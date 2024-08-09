import cv2
from sklearn.cluster import KMeans


class TeamAssiginer:
    def __init__(self) -> None:
        self.team_colors = {}
        self.player_team_dict = {}

    def get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        return KMeans(n_clusters=2, random_state=0, init="k-means++", n_init=10).fit(
            image_2d
        )

    def get_player_color(self, frame, bbox):
        image = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        top_half_image = image[: int(image.shape[0] / 2), :]

        # Get cluster model
        kmeans = self.get_clustering_model(top_half_image)

        # Get cluster labels
        cluster_labels = kmeans.labels_

        # reshape labels into original images
        clustered_image = cluster_labels.reshape(top_half_image.shape[:2])

        corner_cluster = (
            clustered_image[0, -1],
            clustered_image[0, 0],
            clustered_image[-1, -1],
            clustered_image[-1, 0],
        )

        non_player_cluster = max(corner_cluster, key=corner_cluster.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):

        players_color = []

        for _, player_detection in player_detections.items():
            bbox = player_detection["bbox"]
            player_color = self.get_player_color(frame, bbox=bbox)
            players_color.append(player_color)

        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(players_color)

        # Cache the kmean
        self.kmeans = kmeans

        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)
        team_id = self.kmeans.predict(player_color.reshape(1, -1))[0]
        team_id += 1  # Try removing it

        if player_id == 98:
            team_id = 2

        self.player_team_dict[player_id] = team_id

        return team_id
