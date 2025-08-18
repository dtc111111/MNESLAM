import torch
import torch.nn.functional as F
from mp_slam import netvlad

class LoopDetector():
    def __init__(self, config, slam_obj) -> None:
        """
        初始化LoopDetector。

        Args:
            config (dict): 全局配置字典。
            slam_obj: 主SLAM对象，用于访问共享资源如设备、共享描述子数据库等。
        """
        # 从配置中读取闭环检测相关参数，并设置默认值
        loop_config = config.get('loop_detection', {})
        self.loop_launch_th = loop_config.get('loop_launch_th', 20)
        self.min_time_diff = loop_config.get('min_time_diff', 20)
        self.sim_threshold = loop_config.get('sim_threshold', 0.8)
        
        self.device = slam_obj.device

        # 加载预训练的NetVLAD模型
        self.detector = netvlad.NetVLAD(config).eval().to(self.device)

        # 引用来自主SLAM对象的共享描述子数据库和锁
        # 这是实现多智能体共享的关键
        self.descriptor_db = slam_obj.descriptor_db
        self.db_lock = slam_obj.descriptor_db_lock

    def _extract_descriptor(self, frame_rgb):
        """
        从单帧RGB图像中提取NetVLAD描述子。

        Args:
            frame_rgb (torch.Tensor): 形状为 [H, W, 3] 的RGB图像张量，数值范围0-255或0-1。

        Returns:
            torch.Tensor: 形状为 [1, D] 的描述子张量，位于GPU上。
        """
        # 确保图像在正确的设备上并进行归一化
        image = frame_rgb.to(self.device)
        if image.max() > 1.0:
            image = image / 255.0
        
        # 调整维度以匹配模型输入 [B, C, H, W]
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = torch.clamp(image, min=0, max=1)

        with torch.no_grad():
            des = self.detector({'image': image})['global_descriptor']

        return des

    def detect_and_add(self, current_kf_id, current_agent_id, frame_rgb):
        """
        为当前帧提取描述子，检测回环，然后将新描述子添加到数据库。
        这是一个原子操作，简化了Mapper中的逻辑。

        Args:
            current_kf_id (int): 当前关键帧的ID。
            current_agent_id (int): 当前智能体的ID (rank)。
            frame_rgb (torch.Tensor): 当前关键帧的RGB图像 [H, W, 3]。

        Returns:
            dict or None: 如果找到回环，返回包含回环信息的字典，否则返回None。
                          例如: {'match_kf_id': int, 'match_agent_id': int, 'similarity': float}
        """
        # 1. 为当前帧提取描述子
        current_des = self._extract_descriptor(frame_rgb)

        loop_info = None

        # 2. 使用锁保护对共享数据库的访问
        with self.db_lock:
            db_size = len(self.descriptor_db)

            # 3. 如果数据库足够大，则开始检测回环
            if db_size >= self.loop_launch_th:
                # 准备数据库中的数据以供比较
                # 从共享列表中解包描述子、关键帧ID和智能体ID
                db_items = list(self.descriptor_db)
                db_descriptors = [item['descriptor'] for item in db_items]
                db_kf_ids = [item['kf_id'] for item in db_items]
                db_agent_ids = [item['agent_id'] for item in db_items]

                # 将所有候选描述子拼接成一个张量，并移动到GPU以进行高效计算
                candidate_des_tensor = torch.cat(db_descriptors, dim=0).to(self.device)

                # 计算余弦相似度
                sim_scores = F.cosine_similarity(current_des, candidate_des_tensor)

                # 4. 过滤并寻找最佳匹配
                best_score = -1.0
                best_match_idx = -1

                for i in range(len(sim_scores)):
                    # 条件1: 相似度必须高于阈值
                    if sim_scores[i] < self.sim_threshold:
                        continue

                    # 条件2: 如果是同一个智能体，必须有足够的时间间隔，避免与最近的帧形成回环
                    is_same_agent = (db_agent_ids[i] == current_agent_id)
                    time_diff = abs(current_kf_id - db_kf_ids[i])
                    if is_same_agent and time_diff < self.min_time_diff:
                        continue

                    # 这是一个有效的候选回环，检查它是否是最佳匹配
                    if sim_scores[i] > best_score:
                        best_score = sim_scores[i]
                        best_match_idx = i

                # 5. 如果找到了有效的回环，准备返回信息
                if best_match_idx != -1:
                    loop_info = {
                        'match_kf_id': db_kf_ids[best_match_idx],
                        'match_agent_id': db_agent_ids[best_match_idx],
                        'similarity': best_score.item()
                    }

            # 6. 将当前帧的新描述子添加到数据库中
            # 将描述子存储在CPU
            new_db_entry = {
                'descriptor': current_des.cpu(),
                'kf_id': current_kf_id,
                'agent_id': current_agent_id
            }
            self.descriptor_db.append(new_db_entry)

        return loop_info

   
        
