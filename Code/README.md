Student ID: 210877245

We have created 3 new files in 'players/newmcts/' directory as named below:

NewMCTSParams.java     : Class for defining parameters of newMCTS player
NewMCTSPlayer.java     : Class for defining the newMCTS player
NewSingleTreeNode.java : Class for implementing tree search for newMCTS player
-----------------------------------------------
To execute Run.java, follow the below set of instructions:

* Importing packages:
  import players.newmcts.NewMCTSParams;
  import players.newmcts.NewMCTSPlayer;
* Creating the new agent:
  NewMCTSParams newMCTSParams = new NewMCTSParams(); 		     // constructor
  newMCTSParams.stop_type = newMCTSParams.STOP_ITERATIONS;  // stop_type
  newMCTSParams.num_iterations = 200; 						         // number of iterations
  newMCTSParams.rollout_depth = 12;  						          // rollout depth
