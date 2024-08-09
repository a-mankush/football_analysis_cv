from utils import get_center_of_bbox, measure_distance


class PlayerBallAssiginer:
    def __init__(self) -> None:
        self.max_player_ball_distance = 70  # Pixels

    def assign_ball_to_player(self, player, ball_position):
        ball_position = get_center_of_bbox(ball_position)

        minimum_distance = 9999
        assigned_player = -1

        for player_id, player_bbox in player.items():
            distance_left = measure_distance(
                (player_bbox["bbox"][0], player_bbox["bbox"][-1]), ball_position
            )
            distance_right = measure_distance(
                (player_bbox["bbox"][2], player_bbox["bbox"][-1]), ball_position
            )

            distance = min(distance_left, distance_right)

            if distance < self.max_player_ball_distance:
                if distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

        return assigned_player
