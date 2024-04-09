Ultimate frisbee is a sport that has only recently started to gain traction. As a competitive player, I've wanted to develop novel strategies, and, after learning about recent advancements in machine learning, I decided to uncover some of these strategies with AI.

One common approach to solving these types of problems is to utilize a type of algorithm that learns from trial and error. After interacting enough with its environment, this type of algorithm can learn to predict—based on the information it’s given—the best course of action. For ultimate frisbee, the AI would need to use information about itself, information about other players, and the location of the frisbee to predict where it should move or where it should throw the frisbee.

![image](https://github.com/camtamsox/MovingSofa/assets/109252429/78205ae2-b09b-448a-9dd8-9e0de48c206e)
A visualization of the inputs and outputs of the algorithm.

In my program, I decided to first use a simplified version of ultimate frisbee and then work up to a more realistic version as the AI learned more. In this simplification, the frisbee did not spend any time in the air, the AI was always accurate, and the defense did not have any effect on the game. This means that the only way the offense could lose is if they held the frisbee for too long or threw it to someone out of bounds. To predict where it should move or throw the frisbee, the algorithm (a neural network) was given the position of each player and which player had the frisbee.

![image](https://github.com/camtamsox/MovingSofa/assets/109252429/9ba88f68-63f3-47c4-8f07-6c046674914d)

A depiction of the neural network of the AI.

After I got the code to work, I trained the algorithm for thousands of games. However, even after experimenting with different hyperparameters, I could not get the algorithm to learn anything meaningful. A typical game is shown below.

https://github.com/camtamsox/MovingSofa/assets/109252429/a72006ea-f5f4-46e9-a72a-6bd088b0e130


Why this project didn't work:

- The AI had to cooperate with the other six AI players on its team. Cooperation makes learning hard because the AI's reward isn't completely in its own control. This means that if one teammate makes a mistake, everyone is punished so the AI will try to change its behavior even though it isn't at fault.

- There were a lot of actions to choose from. The AI had to select one of the eight directions to move and one of the other six players to throw the frisbee to. Experimenting with all of the actions through trial and error would probably take a lot of time before the AI could understand the effectiveness of each action.

- Training an algorithm from scratch made the learning process more difficult. This is because the AI had no prior knowledge about anything so it had a lot it had to learn. If I could do this project again, I would try to begin with an already trained algorithm for a sport like soccer or football. Also, I could first train the AI on data from existing frisbee games (supervised learning) and then have it train against other AI (reinforcement learning).
