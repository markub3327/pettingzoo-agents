# PettingZoo Agents

## Rewards

The game uses the following reward scheme:

| Outcome | Reward |
|----------|----------|
| Winner | +1 |
| Loser | -1 |
| Draw | 0 for both players |

## Status

| Status | Meaning |
|---------|---------|
| 0 | Running |
| 1 | Started |
| 2 | Win |
| 3 | Lose |
| 4 | Draw |

- **0 (Running)**: The process is currently active and in progress.
- **1 (Started)**: The process has been initiated and is preparing to run.
- **2 (Win)**: The process completed successfully with a winning outcome.
- **3 (Lose)**: The process completed with a losing or unsuccessful outcome.
- **4 (Draw)**: The process completed without a winner or loser, resulting in a tie.