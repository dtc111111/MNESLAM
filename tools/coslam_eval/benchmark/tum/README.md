# TUM-RGBD


| Methods   | ATE↓<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param |
|-----------|:-------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:------:|
| iMAP      |     4.23      |           -           |          -          |        -        |       -       |   -    |
| iMAP*     |     6.10      |       29.6x200        |      44.3x300       |      0.17       |     0.08      |  0.2M  |
| NICE-SLAM |     2.50      |       47.1x200        |      189.2x60       |      0.11       |     0.09      | 101.6M |
| Co-SLAM   |     2.40      |        7.5x10         |       19.0x20       |      13.33      |     2.63      |  1.6M  |
| Co-SLAM†  |     2.17      |        7.5x20         |       19.0x20       |      6.67       |     2.63      |  1.6M  |

The runtime is measured as the total processing time from the program starts until the final checkpoint is saved.
Mesh saving was disabled for NICE-SLAM and iMAP* as we observed that they can take up to ~10 minutes to extract a single mesh.
Also note that run-time of NICE-SLAM and iMAP* on fr2/xyz and fr3/office are approximate estimated values as they take too long to run.

### fr1/desk

| Methods   | ATE↓<br/>[cm] | runtime↓ | #param↓ |
|-----------|:-------------:|:--------:|:-------:|
| iMAP      |      4.9      |    -     |    -    |
| iMAP*     |      7.2      | 3hr17min |  0.22M  |
| NICE-SLAM |      2.7      | 2hr30min | 44.66M  |
| Co-SLAM   |      2.7      | 1min18s  |  1.59M  |
| Co-SLAM†  |      2.4      | 1min56s  |  1.59M  |


### fr2/xyz

| Methods   | ATE↓<br/>[cm] | runtime↓ | #param↓ |
|-----------|:-------------:|:--------:|:-------:|
| iMAP      |      2.0      |    -     |    -    |
| iMAP*     |      2.1      |  ~18hr   |  0.22M  |
| NICE-SLAM |      1.8      |  ~24hr   | 120.99M |
| Co-SLAM   |      1.9      | 6min58s  |  1.59M  |
| Co-SLAM†  |      1.7      | 10min44s |  1.59M  |

### fr3/office


| Methods   | ATE↓<br/>[cm] | runtime↓ | #param↓ |
|-----------|:-------------:|:--------:|:-------:|
| iMAP      |      5.8      |    -     |    -    |
| iMAP*     |      9.0      |  ~15hr   |  0.22M  |
| NICE-SLAM |      3.0      |  ~16hr   | 139.19M |
| Co-SLAM   |      2.6      | 5min33s  |  1.59M  |
| Co-SLAM†  |      2.4      | 8min23s  |  1.59M  |
