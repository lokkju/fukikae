"""Identity evaluation harness: segment episodes, embed the segments once into a cache, replay
the registry's episode-by-episode matching in memory and score cross-episode identity against
subtitle ground truth. Never writes the registry or any pipeline artifact."""
