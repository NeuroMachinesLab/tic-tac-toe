package ai.neuromachines.tictactoe;

import lombok.SneakyThrows;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

public class QTable {
    private static final float MIN_REWARD = -10;
    private final Map<BoardState, float[]> rewards = new LinkedHashMap<>();

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
        float[] r = getRewards(values);
        this.rewards.put(state, r);
    }

    private static float[] getRewards(String[] values) {
        float[] rewards = new float[9];
        for (int i = 1; i < values.length; i++) {
            String s = values[i].trim();
            rewards[i - 1] = Objects.equals(s, "-") ?
                    MIN_REWARD :
                    Float.parseFloat(s);
        }
        return rewards;
    }

    public Set<BoardState> getStates() {
        return rewards.keySet();
    }

    public int getMaxRewardAction(BoardState state) {
        float[] r = rewards.get(state);
        return argMax(r);
    }

    private static int argMax(float[] array) {
        int maxI = 0;
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
                maxI = i;
            }
        }
        return maxI;
    }
}
