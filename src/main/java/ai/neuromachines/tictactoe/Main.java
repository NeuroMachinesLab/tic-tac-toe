import ai.neuromachines.file.NetworkSerializer;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.train.TrainStrategy;
import ai.neuromachines.tictactoe.BoardState;
import ai.neuromachines.tictactoe.QTable;

import static java.lang.IO.print;
import static java.lang.IO.println;
import static java.nio.file.StandardOpenOption.*;

void main() throws IOException {
    // Trained network file (it may be missing)
    Path path = Path.of("network.txt");

    // Read Q-Learning Tables
    QTable qTable = new QTable();
    qTable.readFile("q-table-x.csv");
    qTable.readFile("q-table-o.csv");

    // Build Network
    Network network = Files.exists(path) ?
            openNetworkFromFile(path) :
            createNetwork(9, 64, 9);

    Network bestNetwork = trainNetwork(network, qTable, 1000);

    saveToFile(bestNetwork, path);
}

Network createNetwork(int... layersNodeCount) {
    println("Create network with random weights and " +
            layersNodeCount[0] + " nodes in input layer, " +
            layersNodeCount[1] + " nodes in hidden layer, " +
            layersNodeCount[2] + " nodes in output layer");
    ActivationFunc actFuncHidden = ActivationFunc.sigmoid(0.6f);  // found empirically
    ActivationFunc actFuncOutput = ActivationFunc.softmax();
    return Network.of(List.of(actFuncHidden, actFuncOutput), layersNodeCount);
}

@SuppressWarnings("SameParameterValue")
Network openNetworkFromFile(Path path) throws IOException {
    println("Read network from: " + path);
    try (FileChannel ch = FileChannel.open(path)) {
        return NetworkSerializer.deserialize(ch);
    }
}

@SuppressWarnings("SameParameterValue")
void saveToFile(Network network, Path path) throws IOException {
    try (FileChannel ch = FileChannel.open(path, CREATE, WRITE, TRUNCATE_EXISTING)) {
        NetworkSerializer.serialize(network, ch);
    }
    println("The most accurate network from the iterations has been written to: " + path);
}

/**
 * @return the most accurate network from the iterations
 */
@SuppressWarnings("SameParameterValue")
Network trainNetwork(Network network, QTable qtable, int iterations) {
    println("Train iterations: " + iterations);
    Instant t0 = Instant.now();

    TrainStrategy trainStrategy = TrainStrategy.backpropagation(network);
    trainStrategy.setLearningRate(0.02f);  // found empirically

    Network bestNetwork = Network.of(network);
    int fewestMistakes = quietTestNetwork(bestNetwork, qtable);

    for (int i = 1; i <= iterations; i++) {
        trainNetwork(qtable, trainStrategy);
        int mistakes = quietTestNetwork(network, qtable);
        if (mistakes < fewestMistakes) {
            fewestMistakes = mistakes;
            bestNetwork = Network.of(network);
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

void trainNetwork(QTable qtable, TrainStrategy trainStrategy) {
    for (BoardState state : qtable.getStates()) {
        float[] input = state.getNetworkInput();
        float[] expectedOutput = getExpectedSoftmaxOutput(state, qtable);

        trainStrategy.train(input, expectedOutput);
    }
}

float[] getExpectedSoftmaxOutput(BoardState state, QTable qTable) {
    Set<Integer> preferredActions = qTable.getMaxRewardActions(state);
    float value = 1.0f / preferredActions.size();
    float[] output = new float[9];
    for (int i : preferredActions) {
        output[i] = value;
    }
    return output;
}

int quietTestNetwork(Network network, QTable qtable) {
    return testNetwork(network, qtable, true);
}

int testNetworkAndPrintResult(Network network, QTable qtable) {
    return testNetwork(network, qtable, false);
}

int testNetwork(Network network, QTable qtable, boolean quiet) {
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

void printResult(BoardState state, Collection<Integer> preferredMoves, Collection<Integer> predictedMoves, boolean hasMistakes) {
    System.out.printf("%s : best move(-s) (any of) = %-27s, network answer  (any of) = %s",
            state, preferredMoves, predictedMoves);
    if (hasMistakes) {
        print("\t[MISTAKE]");
    }
    println();
}