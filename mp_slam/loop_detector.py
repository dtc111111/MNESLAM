import torch
import torch.nn.functional as F
from mp_slam import netvlad

class LoopDetector():
    def __init__(self, config, slam_obj) -> None:
        loop_config = config.get('loop_detection', {})
        self.loop_launch_th = loop_config.get('loop_launch_th', 20)
        self.min_time_diff = loop_config.get('min_time_diff', 20)
        self.sim_threshold = loop_config.get('sim_threshold', 0.8)
        self.device = slam_obj.device
        self.detector = netvlad.NetVLAD(config).eval().to(self.device)
        self.descriptor_db = slam_obj.descriptor_db
        self.db_lock = slam_obj.descriptor_db_lock

    def _extract_descriptor(self, frame_rgb):
        image = frame_rgb.to(self.device)
        if image.max() > 1.0:
            image = image / 255.0
        
        image = image.permute(2, 0, 1).unsqueeze(0)
        image = torch.clamp(image, min=0, max=1)

        with torch.no_grad():
            des = self.detector({'image': image})['global_descriptor']

        return des

    def detect_and_add(self, current_kf_id, current_agent_id, frame_rgb):
        current_des = self._extract_descriptor(frame_rgb)

        loop_info = None

        with self.db_lock:
            db_size = len(self.descriptor_db)

            if db_size >= self.loop_launch_th:
                db_items = list(self.descriptor_db)
                db_descriptors = [item['descriptor'] for item in db_items]
                db_kf_ids = [item['kf_id'] for item in db_items]
                db_agent_ids = [item['agent_id'] for item in db_items]

                candidate_des_tensor = torch.cat(db_descriptors, dim=0).to(self.device)

                sim_scores = F.cosine_similarity(current_des, candidate_des_tensor)

                best_score = -1.0
                best_match_idx = -1

                for i in range(len(sim_scores)):
                    if sim_scores[i] < self.sim_threshold:
                        continue

                    is_same_agent = (db_agent_ids[i] == current_agent_id)
                    time_diff = abs(current_kf_id - db_kf_ids[i])
                    if is_same_agent and time_diff < self.min_time_diff:
                        continue
                    if sim_scores[i] > best_score:
                        best_score = sim_scores[i]
                        best_match_idx = i

                if best_match_idx != -1:
                    loop_info = {
                        'match_kf_id': db_kf_ids[best_match_idx],
                        'match_agent_id': db_agent_ids[best_match_idx],
                        'similarity': best_score.item()
                    }

            new_db_entry = {
                'descriptor': current_des.cpu(),
                'kf_id': current_kf_id,
                'agent_id': current_agent_id
            }
            self.descriptor_db.append(new_db_entry)

        return loop_info