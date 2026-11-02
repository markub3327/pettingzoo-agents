import datetime
import cv2
import numpy as np
from datasets import load_from_disk
from collections import Counter

### PettingZoo
env_names = {
    "classic/tictactoe-v3": {"color": "white"},
    "classic/rps-v2": {"color": "black"},
}


for env_name in env_names.keys():
    # Load the dataset from disk
    ds = load_from_disk(f"./dataset/{env_name.replace("/", "_")}")
    print(f"Size of dataset for {env_name}: {len(ds)}")
    print(f"Features of dataset for {env_name}: {ds.features}\n")

    # Get the count of samples for each status in the dataset
    status_counts = Counter(row['messages']['status'] for row in ds)
    print(f"Status counts: {status_counts}")

    # Get the count of samples for each reward in the dataset
    reward_counts = Counter(row['messages']['reward'] for row in ds)
    print(f"Reward counts: {reward_counts}\n")

    # Create a video writer object to save the frames as a video
    video_out = cv2.VideoWriter(
        filename=f"./videos/{env_name.replace("/", "_")}_{datetime.datetime.now().strftime('%d-%m-%Y_%H-%M-%S')}.mp4",
        fourcc=cv2.VideoWriter_fourcc(*'mp4v'),
        fps=25,
        frameSize=(1000, 1000),
    )

    for row in ds:
        # Print the current agent's name
        print(f"Observation:\n{row['messages']['observation']}")
        print(f"Action Mask:\n{row['messages']['action_mask']}")

        for frame_id, img in enumerate(row['images']):
            # Convert to BGR format for OpenCV
            frame = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

            # Resize the frame
            frame = cv2.resize(frame, (1000, 1000), interpolation=cv2.INTER_LANCZOS4)

            # Add text to the frame
            cv2.putText(
                frame,
                f"Game: {row['messages']['game']}",
                (40, 70),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 18, 25) if env_names[env_name]["color"] == "black" else (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Agent: {row['messages']['name']}",
                (40, 100),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 18, 25) if env_names[env_name]["color"] == "black" else (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Action: {row['messages']['action']}",
                (40, 130),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 18, 25) if env_names[env_name]["color"] == "black" else (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Reward: {row['messages']['reward']}",
                (40, 160),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 18, 25) if env_names[env_name]["color"] == "black" else (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"Status: {row['messages']['status']}",
                (40, 190),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 18, 25) if env_names[env_name]["color"] == "black" else (234, 232, 233),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                frame,
                f"FrameID: {frame_id}",
                (40, 220),
                cv2.FONT_HERSHEY_DUPLEX,
                0.8,
                (0, 18, 25) if env_names[env_name]["color"] == "black" else (234, 232, 233),
                2,
                cv2.LINE_AA,
            )

            # Write the frame to the video
            video_out.write(frame)

    video_out.release()