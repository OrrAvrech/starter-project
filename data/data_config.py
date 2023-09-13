from dataclasses import dataclass


@dataclass
class ScraperConfig:
    extractor: str
    prefix_prompt: str
    restrict_filenames: bool
    min_vid_duration: int
    max_vid_duration: int
    ext: str
    no_playlist: bool
    desired_agg_duration: int
    quiet_mode: bool


@dataclass
class DataConfig:
    scraper: ScraperConfig
    actions: list[str]
