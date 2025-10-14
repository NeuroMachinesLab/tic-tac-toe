import ai.neuromachines.file.NetworkSerializer;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.network.train.TrainStrategy;
import ai.neuromachines.tictactoe.BoardState;
import ai.neuromachines.tictactoe.QTable;

import static java.lang.IO.println;
import static java.nio.file.StandardOpenOption.*;

static int TRAIN_ITERATIONS = 3000;

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

    trainNetwork(network, qTable, TRAIN_ITERATIONS);

    saveToFile(network, path);
}

Network createNetwork(int... layersNodeCount) {
    println("Create network with random weights and " +
            layersNodeCount[0] + " nodes in input layer, " +
            layersNodeCount[1] + " nodes in hidden layer, " +
            layersNodeCount[2] + " nodes in output layer");
    ActivationFunc actFuncHidden = ActivationFunc.sigmoid(0.0002f * TRAIN_ITERATIONS);  // found empirically
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
    println("Network has been written to: " + path);
}

@SuppressWarnings("SameParameterValue")
void trainNetwork(Network network, QTable qtable, int iterations) {
    println("Train iterations: " + iterations);
    Instant t0 = Instant.now();
    TrainStrategy trainStrategy = TrainStrategy.backpropagation(network);
    trainStrategy.setLearningRate(20f / iterations);  // found empirically
    int percentDecate = iterations / 10;
    for (int i = 1; i <= iterations; i++) {
        trainNetwork(qtable, trainStrategy);
        if (iterations >= 100 && (i % percentDecate) == 0) {
            println(10 * i / percentDecate + "% done");
        }
    }
    Duration timeSpent = Duration.between(t0, Instant.now());
    int incorrectMoveCnt = testNetwork(network, qtable);
    println("Trained for " + qtable.getStates().size() + " states");
    println("Warnings: " + incorrectMoveCnt);
    System.out.printf("Accuracy: %.1f%%\n", 100 - (100.0 * incorrectMoveCnt / qtable.getStates().size()));
    println("Train time: " + timeSpent);
}

static void trainNetwork(QTable qtable, TrainStrategy trainStrategy) {
    for (BoardState state : qtable.getStates()) {
        float[] input = state.getNetworkInput();
        float[] expectedOutput = getExpectedSoftmaxOutput(state, qtable);

        trainStrategy.train(input, expectedOutput);
    }
}

static float[] getExpectedSoftmaxOutput(BoardState state, QTable qTable) {
    List<Integer> prefferedActions = qTable.getMaxRewardActions(state);
    float value = 1.0f / prefferedActions.size();
    float[] output = new float[9];
    for (int i : prefferedActions) {
        output[i] = value;
    }
    return output;
}

int testNetwork(Network network, QTable qtable) {
    int incorrectCnt = 0;
    for (BoardState state : qtable.getStates()) {
        List<Integer> preferredMoves = qtable.getMaxRewardActions(state);
        float[] input = state.getNetworkInput();

        network.input(input);
        network.propagate();

        float[] output = network.output();
        int predictedMove = QTable.argMax(output);
        boolean isCorrectMove = printResult(state, preferredMoves, predictedMove);
        if (!isCorrectMove) {
            incorrectCnt++;
        }
    }
    return incorrectCnt;
}

boolean printResult(BoardState state, List<Integer> possibleMoves, int answer) {
    System.out.printf("%s : expected one of = %-27s, network answer = %d", state, possibleMoves, answer);
    boolean isCorrectMove = possibleMoves.contains(answer);
    if (isCorrectMove) {
        println();
    } else {
        println("\t[WARNING]");
    }
    return isCorrectMove;
}