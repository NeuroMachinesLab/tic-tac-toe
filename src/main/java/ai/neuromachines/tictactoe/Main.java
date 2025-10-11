import ai.neuromachines.file.NetworkSerializer;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.train.TrainStrategy;
import ai.neuromachines.tictactoe.BoardState;
import ai.neuromachines.tictactoe.QTable;

import static java.lang.IO.println;
import static java.nio.file.StandardOpenOption.*;

/**
 * Network output contains only one value and belongs to 0..1 interval.
 * Network trained better for 0..1 output interval.<p>
 * Multiply output to 10 to convert it to Tic Tac Board Cell Index.
 * This index belongs to 0..8 interval (0 for upper left, 8 for lower right cell).
 */
float Multiplicator = 10;

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
            createNetwork(9, 1);

    // Train
    trainNetwork(network, qTable, 1000);

    // Check network
    testNetwork(network, qTable);

    saveToFile(network, path);
}

Network createNetwork(int... layersNodeCount) {
    println("Create network with random weights and " +
            layersNodeCount[0] + " nodes in input layer, " +
            layersNodeCount[1] + " nodes in output layer");
    ActivationFunc actFunc = ActivationFunc.tanh();
    return Network.of(List.of(actFunc, actFunc), layersNodeCount);
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
    println("Network has been written to: " + path);
}

@SuppressWarnings("SameParameterValue")
private void trainNetwork(Network network, QTable qtable, int iterations) {
    println("Train iterations: " + iterations);
    Instant t0 = Instant.now();
    TrainStrategy trainStrategy = TrainStrategy.backpropagation(network);
    for (BoardState state : qtable.getStates()) {
        int move = qtable.getMaxRewardAction(state);
        float[] input = state.getNetworkInput();
        float[] expectedOutput = new float[1];
        expectedOutput[0] = move / Multiplicator;

        trainNetwork(input, expectedOutput, trainStrategy, iterations);
    }
    Duration timeSpent = Duration.between(t0, Instant.now());
    println("Trained for " + qtable.getStates().size() + " states");
    println("Train time: " + timeSpent);
}

@SuppressWarnings("SameParameterValue")
void trainNetwork(float[] input, float[] expectedOutput, TrainStrategy trainStrategy, int iterations) {
    for (int i = 0; i < iterations; i++) {
        trainStrategy.train(input, expectedOutput);
    }
}

void testNetwork(Network network, QTable qtable) {
    for (BoardState state : qtable.getStates()) {
        int move = qtable.getMaxRewardAction(state);
        float[] input = state.getNetworkInput();

        network.input(input);
        network.propagate();

        float networkOutput = network.output()[0];
        float predictedMove = networkOutput * Multiplicator;
        printResult(state, move, predictedMove);
    }
}

void printResult(BoardState state, int expected, float answer) {
    float error = expected - answer;
    System.out.printf("%s : expected = %d,\t\tnetwork answer = %+.2f,\t\terror = %+.0e", state, expected, answer, error);
    if (Math.abs(error) > 0.01f) {
        println("\t[WARNING]");
    } else {
        println();
    }
}