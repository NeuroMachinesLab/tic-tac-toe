package ai.neuromachines.tictactoe;

import lombok.RequiredArgsConstructor;

public record BoardState(State[] state) {

    /**
     * @param description in "--- xxx ooo" format
     */
    public BoardState(String description) {
        State[] state = new State[9];
        int i = 0;
        for (char c : description.toCharArray()) {
            if (c == '-') {
                state[i++] = State.BLANK;
            } else if (c == 'o') {
                state[i++] = State.O;
            } else if (c == 'x') {
                state[i++] = State.X;
            }
        }
        if (i != 9) {
            throw new IllegalArgumentException("Illegal board description");
        }
        this(state);
    }

    public float[] getNetworkInput() {
        float[] input = new float[state.length];
        int i = 0;
        for (State s : state) {
            input[i++] = s.state;
        }
        return input;
    }

    @Override
    public String toString() {
        int i = 0;
        StringBuilder sb = new StringBuilder(11);
        for (State s : state) {
            switch (s) {
                case BLANK -> sb.append('-');
                case O -> sb.append('o');
                case X -> sb.append('x');
            }
            i++;
            if (i == 3 || i == 6) {
                sb.append(' ');
            }
        }
        return sb.toString();
    }

    @RequiredArgsConstructor
    public enum State {
        BLANK(-1),
        O(0),
        X(1);

        private final int state;
    }
}
