package ai.neuromachines.tictactoe;

import lombok.Getter;
import lombok.SneakyThrows;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

@Getter
public class QTable {
    private final Map<BoardState, Integer> moves = new LinkedHashMap<>();

    @SneakyThrows
    public void readFile(String fileName) {
        Path path = Paths.get(fileName);
        try (BufferedReader reader = Files.newBufferedReader(path)) {
            reader.readLine();  // skip header
            String line;
            while ((line = reader.readLine()) != null) {
                parseLine(line);
            }
        }
    }

    private void parseLine(String line) {
        String[] values = line.split(",");
        BoardState state = new BoardState(values[0]);
        Integer move = getCellToMove(values);
        moves.put(state, move);
    }

    /**
     * @return zero-based board cell index to move
     */
    private static int getCellToMove(String[] values) {
        double max = Double.NEGATIVE_INFINITY;
        int maxIdx = -1;
        for (int i = 1; i < values.length; i++) {
            String s = values[i].trim();
            if (!Objects.equals(s, "-")) {
                double v = Double.parseDouble(s);
                if (v > max) {
                    max = v;
                    maxIdx = i;
                }
            }
        }
        if (maxIdx == -1) {
            throw new IllegalArgumentException("Incorrect Q_learning Table row values");
        }
        return maxIdx - 1;
    }
}
