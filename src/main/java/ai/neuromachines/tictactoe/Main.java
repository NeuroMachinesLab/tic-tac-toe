import ai.neuromachines.file.NetworkSerializer;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.train.TrainStrategy;
import ai.neuromachines.tictactoe.BoardState;
import ai.neuromachines.tictactoe.QTable;

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
            createNetwork(9, 5, 1);

    // Train
    trainNetwork(network, qTable, 100);

    saveToFile(network, path);
}

Network createNetwork(int... layersNodeCount) {
    println("Create network with random weights and " +
            layersNodeCount[0] + " nodes in input layer, " +
            layersNodeCount[1] + " nodes in hidden layer, " +
            layersNodeCount[2] + " nodes in output layer");
    ActivationFunc actFunc = ActivationFunc.leakyReLu(0.01f);
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
    for (var e : qtable.getMoves().entrySet()) {
        BoardState state = e.getKey();
        Integer move = e.getValue();
        float[] input = state.getNetworkInput();
        float[] expectedOutput = new float[1];
        expectedOutput[0] = move;

        trainNetwork(input, expectedOutput, trainStrategy, iterations);
        printResult(state, move, network);
    }
    Duration timeSpent = Duration.between(t0, Instant.now());
    println("Train time: " + timeSpent);
}

@SuppressWarnings("SameParameterValue")
void trainNetwork(float[] input, float[] expectedOutput, TrainStrategy trainStrategy, int iterations) {
    for (int i = 0; i < iterations; i++) {
        trainStrategy.train(input, expectedOutput);
    }
}

void printResult(BoardState state, int expect, Network network) {
    float answer = network.output()[0];
    println(state + ": expected = " + expect + ",\tnetwork answer = " + answer + ",\terror = " + (expect - answer));
}