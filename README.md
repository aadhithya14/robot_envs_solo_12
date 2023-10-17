# Environment for Solo12

Environment for learning robust goal conditioned agile locomotion policies on 12DOF quadruped robot Solo12 and Sim2Real transfer to the Real Robot.

The learning comprises of multiple stages. 

[x] The first stage includes offline imitation of the demonstration using PPO with imitation reward and random initialization at any point in demonstration trajectory. 

[x]The next stage involves discarding the demonstration and online learning on top of the base policy learnt using imitation. This stage involves task specific rewards, trajectory tracking reward for ensuring the robustness of the PD controller and torque smoothness penalties for easy transfer to the real robot.

### Results on the Real Robot





