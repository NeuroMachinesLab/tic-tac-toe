package ai.neuromachines.tictactoe;

import lombok.SneakyThrows;

import java.io.BufferedReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

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

    public Collection<BoardState> getStates() {
        List<BoardState> states = new ArrayList<>(rewards.keySet());
        Collections.shuffle(states);
        return states;
    }

    public Set<Integer> getMaxRewardActions(BoardState state) {
        float[] r = rewards.get(state);
        return getMaxRewardActions(r);
    }

    public static Set<Integer> getMaxRewardActions(float[] rewards) {
        float max = max(rewards);
        return findValuesEqualsTo(rewards, max);
    }

    private static Set<Integer> findValuesEqualsTo(float[] a, float v) {
        Set<Integer> maxArgs = new HashSet<>(9);
        for (int i = 0; i < a.length; i++) {
            if (a[i] == v) {
                maxArgs.add(i);
            }
        }
        return maxArgs;
    }

    private static float max(float[] array) {
        float max = array[0];
        for (int i = 1; i < array.length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
        return max;
    }
}
