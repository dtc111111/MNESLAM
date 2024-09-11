# SyntheticRGBD

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | Track.↓<br/>[ms x it] | Map.↓<br/>[ms x it] | Track.<br/>FPS↑ | Map.<br/>FPS↑ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:---------------------:|:-------------------:|:---------------:|:-------------:|:-------:|
|   iMAP*    |     18.29     |     26.41      |          20.73          |          47.22          |        31.0x50        |      49.1x300       |      0.64       |     0.07      |  0.22M  |
| NICE-SLAM  |     5.95      |      5.30      |          77.46          |          6.32           |        12.3x10        |       50.4x60       |      8.13       |     0.33      |  3.11M  |
| Vox-Fusion |     4.10      |      4.81      |          81.78          |          6.13           |        16.6x30        |       46.2x10       |      2.00       |     2.16      |  0.84M  |
|  Co-SLAM   |     2.95      |      2.96      |          86.88          |          3.02           |        6.4x10         |       10.4x10       |      15.63      |     9.62      |  0.26M  |

The runtime is measured as the total processing time from the program starts until the final checkpoint is saved.
Mesh saving was disabled for NICE-SLAM and iMAP* as we observed that they can take up to ~10 minutes to extract a single mesh.

### breakfast room

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | runtime↓ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:--------:|:-------:|
|   iMAP*    |     10.56     |     11.27      |          46.91          |          24.03          | 1hr25min |  0.22M  |
| NICE-SLAM  |     3.44      |      3.69      |          87.69          |          3.66           | 16min15s |  3.40M  |
| Vox-Fusion |     2.23      |      3.02      |          89.67          |          4.79           | 15min5s  |  0.80M  |
|  Co-SLAM   |     1.97      |      1.93      |          94.75          |          3.51           | 1min51s  |  0.26M  |


### complete kitchen

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | runtime↓ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:--------:|:-------:|
|   iMAP*    |     25.16     |     31.09      |          12.96          |          63.59          | 1hr28min |  0.22M  |
| NICE-SLAM  |     10.92     |     12.00      |          55.41          |          12.08          | 13min42s |  7.33M  |
| Vox-Fusion |     3.66      |     13.53      |          66.16          |          14.66          | 15min40s |  1.00M  |
|  Co-SLAM   |     4.68      |      4.94      |          68.91          |          5.62           | 1min58s  |  0.26M  |

### green room

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | runtime↓ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:--------:|:-------:|
|   iMAP*    |     13.01     |     19.17      |          21.78          |          26.22          | 1hr44min |  0.22M  |
| NICE-SLAM  |     5.34      |      4.94      |          82.78          |          10.88          | 12min44s |  3.05M  |
| Vox-Fusion |     14.58     |      4.19      |          74.31          |          5.78           | 15min20s |  0.85M  |
|  Co-SLAM   |     2.10      |      2.96      |          90.80          |          1.95           | 2min15s  |  0.26M  |


### grey white room

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | runtime↓ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:--------:|:-------:|
|   iMAP*    |     11.90     |     20.39      |          20.48          |          21.32          | 1hr48min |  0.22M  |
| NICE-SLAM  |     2.63      |      3.15      |          87.72          |          2.57           | 14min2s  |  2.40M  |
| Vox-Fusion |     2.13      |      2.78      |          92.06          |          2.26           | 15min40s |  0.85M  |
|  Co-SLAM   |     1.89      |      2.16      |          95.04          |          1.25           | 2min22s  |  0.26M  |

### morning apartment

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | runtime↓ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:--------:|:-------:|
|   iMAP*    |     29.62     |     49.22      |          10.72          |          61.29          | 1hr6min  |  0.22M  |
| NICE-SLAM  |     6.55      |      3.13      |          85.04          |          1.72           | 8min19s  |  2.00M  |
| Vox-Fusion |     1.79      |      3.06      |          86.38          |          2.66           |   8min   |  0.75M  |
|  Co-SLAM   |     1.60      |      2.67      |          86.98          |          1.41           | 1min24s  |  0.26M  |

### thin geometry

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | runtime↓ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:--------:|:-------:|
|   iMAP*    |     12.98     |     21.07      |          19.17          |          29.16          | 29min40s |  0.22M  |
| NICE-SLAM  |     3.57      |      5.28      |          72.05          |          7.74           | 3min43s  |  0.38M  |
| Vox-Fusion |     1.46      |      3.15      |          83.01          |          5.28           | 3min40s  |  0.73M  |
|  Co-SLAM   |     3.38      |      2.74      |          86.74          |          4.66           |   37s    |  0.26M  |

### white room

|  Methods   | Acc↓<br/>[cm] | Comp↓<br/>[cm] | Comp<br/>Ratio↑<br/>[%] | Depth <br/>L1↓<br/>[cm] | runtime↓ | #param↓ | 
|:----------:|:-------------:|:--------------:|:-----------------------:|:-----------------------:|:--------:|:-------:|
|   iMAP*    |     24.82     |     32.63      |          13.07          |          81.71          | 1hr59min |  0.22M  |
| NICE-SLAM  |     9.22      |      4.89      |          71.56          |          5.59           | 17min29s |  3.20M  |
| Vox-Fusion |     2.90      |      3.99      |          80.87          |          7.51           | 20min40s |  0.93M  |
|  Co-SLAM   |     5.03      |      3.34      |          84.94          |          2.74           | 3min40s  |  0.26M  |
