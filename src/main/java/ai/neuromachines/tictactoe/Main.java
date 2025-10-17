import ai.neuromachines.file.NetworkSerializer;
import ai.neuromachines.network.Network;
import ai.neuromachines.network.function.ActivationFunc;
import ai.neuromachines.tictactoe.NetworkTrainer;
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
            createNetwork(9, 64, 9);

    Network bestNetwork = NetworkTrainer.train(network, qTable, 1000);

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
