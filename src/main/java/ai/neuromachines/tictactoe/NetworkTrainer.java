package ai.neuromachines.tictactoe;

import ai.neuromachines.network.Network;
import ai.neuromachines.network.train.TrainStrategy;

import java.time.Duration;
import java.time.Instant;
import java.util.Collection;
import java.util.Set;

import static java.lang.IO.print;
import static java.lang.IO.println;

public class NetworkTrainer {

    /**
     * @return the most accurate network from the iterations
     */
    @SuppressWarnings("SameParameterValue")
    public static Network train(Network network, QTable qtable, int iterations) {
        println("Train iterations: " + iterations);
        Instant t0 = Instant.now();

        TrainStrategy trainStrategy = TrainStrategy.backpropagation(network);
        trainStrategy.setLearningRate(0.02f);  // found empirically

        Network bestNetwork = Network.copyOf(network);
        int fewestMistakes = quietTestNetwork(bestNetwork, qtable);

        for (int i = 1; i <= iterations; i++) {
            trainNetwork(qtable, trainStrategy);
            int mistakes = quietTestNetwork(network, qtable);
            if (mistakes < fewestMistakes) {
                fewestMistakes = mistakes;
                bestNetwork = Network.copyOf(network);
            }

            if (iterations >= 100 && (i % (iterations / 10)) == 0) {
                println(100 * i / iterations + "% done");
            }
        }
        int mistakes = testNetworkAndPrintResult(bestNetwork, qtable);
        println("Trained for " + qtable.getStates().size() + " states");
        println("Mistakes: " + mistakes);
        System.out.printf("Accuracy: %.1f%%\n", 100 - (100.0 * mistakes / qtable.getStates().size()));

        Duration timeSpent = Duration.between(t0, Instant.now());
        println("Train time: " + timeSpent);
        return bestNetwork;
    }

    private static void trainNetwork(QTable qtable, TrainStrategy trainStrategy) {
        for (BoardState state : qtable.getStates()) {
            float[] input = state.getNetworkInput();
            float[] expectedOutput = getExpectedSoftmaxOutput(state, qtable);

            trainStrategy.train(input, expectedOutput);
        }
    }

    private static float[] getExpectedSoftmaxOutput(BoardState state, QTable qTable) {
        Set<Integer> preferredActions = qTable.getMaxRewardActions(state);
        float value = 1.0f / preferredActions.size();
        float[] output = new float[9];
        for (int i : preferredActions) {
            output[i] = value;
        }
        return output;
    }

    private static int quietTestNetwork(Network network, QTable qtable) {
        return testNetwork(network, qtable, true);
    }

    private static int testNetworkAndPrintResult(Network network, QTable qtable) {
        return testNetwork(network, qtable, false);
    }

    private static int testNetwork(Network network, QTable qtable, boolean quiet) {
        int mistakes = 0;
        for (BoardState state : qtable.getStates()) {
            Collection<Integer> preferredMoves = qtable.getMaxRewardActions(state);
            float[] input = state.getNetworkInput();

            network.input(input);
            network.propagate();

            float[] output = network.output();
            Collection<Integer> predictedMoves = QTable.getMaxRewardActions(output);

            boolean hasMistakes = !preferredMoves.containsAll(predictedMoves);
            if (hasMistakes) {
                mistakes++;
            }
            if (!quiet) {
                printResult(state, preferredMoves, predictedMoves, hasMistakes);
            }
        }
        return mistakes;
    }

    private static void printResult(BoardState state, Collection<Integer> preferredMoves, Collection<Integer> predictedMoves, boolean hasMistakes) {
        System.out.printf("%s : best move(-s) (any of) = %-27s, network answer  (any of) = %s",
                state, preferredMoves, predictedMoves);
        if (hasMistakes) {
            print("\t[MISTAKE]");
        }
        println();
    }
}
