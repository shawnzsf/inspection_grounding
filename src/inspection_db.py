#!/usr/bin/env python3
"""
Inspection Database — SQLite wrapper for storing per-object inspection data.

Schema:
    - observations: one row per annotation per timestamp (per-frame)
    - objects:      one row per unique track_id (aggregated across frames)
    - categories:   category lookup table

Usage:
    db = InspectionDB("/path/to/inspection.db")
    db.add_observation(timestamp_ns=..., track_id=1, category="Poster", ...)
    db.upsert_object(track_id=1, category="Poster", ...)
    db.close()
"""

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class InspectionDB:
    """SQLite wrapper for inspection object database."""

    def __init__(self, db_path):
        """
        Open (or create) the inspection database.

        Args:
            db_path: Path to the .db file. Parent directories are created
                     automatically.
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self.init_db()

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def init_db(self):
        """Create tables if they don't exist."""
        cur = self.conn.cursor()

        cur.execute("""
            CREATE TABLE IF NOT EXISTS categories (
                id    INTEGER PRIMARY KEY,
                name  TEXT NOT NULL UNIQUE
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp_ns    INTEGER NOT NULL,
                track_id        INTEGER,
                category        TEXT,
                image_id        INTEGER,
                image_file_name TEXT,
                centroid_x      REAL,
                centroid_y      REAL,
                centroid_z      REAL,
                bbox3d_min_x    REAL,
                bbox3d_min_y    REAL,
                bbox3d_min_z    REAL,
                bbox3d_max_x    REAL,
                bbox3d_max_y    REAL,
                bbox3d_max_z    REAL,
                point_count     INTEGER,
                pcd_path        TEXT,
                mask_path       TEXT,
                bbox_2d         TEXT,
                tf_translation_x REAL,
                tf_translation_y REAL,
                tf_translation_z REAL,
                tf_rotation_x    REAL,
                tf_rotation_y    REAL,
                tf_rotation_z    REAL,
                tf_rotation_w    REAL,
                created_at      TEXT DEFAULT (datetime('now'))
            )
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS objects (
                id                    INTEGER PRIMARY KEY AUTOINCREMENT,
                track_id              INTEGER UNIQUE,
                category              TEXT,
                centroid_x            REAL,
                centroid_y            REAL,
                centroid_z            REAL,
                bbox3d_min_x          REAL,
                bbox3d_min_y          REAL,
                bbox3d_min_z          REAL,
                bbox3d_max_x          REAL,
                bbox3d_max_y          REAL,
                bbox3d_max_z          REAL,
                total_point_count     INTEGER,
                observation_count     INTEGER,
                first_seen_ns         INTEGER,
                last_seen_ns          INTEGER,
                aggregated_pcd_path   TEXT,
                tf_translation_x      REAL,
                tf_translation_y      REAL,
                tf_translation_z      REAL,
                tf_rotation_x         REAL,
                tf_rotation_y         REAL,
                tf_rotation_z         REAL,
                tf_rotation_w         REAL,
                created_at            TEXT DEFAULT (datetime('now'))
            )
        """)

        # ------------------------------------------------------------------
        # Migrations: add tf columns to pre-existing databases (idempotent)
        # ------------------------------------------------------------------
        tf_cols = [
            "tf_translation_x", "tf_translation_y", "tf_translation_z",
            "tf_rotation_x", "tf_rotation_y", "tf_rotation_z", "tf_rotation_w",
        ]
        for table in ("observations", "objects"):
            cur.execute(f"PRAGMA table_info({table})")
            existing = {row[1] for row in cur.fetchall()}
            for col in tf_cols:
                if col not in existing:
                    cur.execute(
                        f"ALTER TABLE {table} ADD COLUMN {col} REAL"
                    )

        # Indexes for common queries
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_track
            ON observations(track_id)
        """)
        cur.execute("""
            CREATE INDEX IF NOT EXISTS idx_obs_timestamp
            ON observations(timestamp_ns)
        """)

        self.conn.commit()

    # ------------------------------------------------------------------
    # Category helpers
    # ------------------------------------------------------------------

    def add_category(self, category_id, name):
        """Insert a category if it doesn't already exist."""
        self.conn.execute(
            "INSERT OR IGNORE INTO categories (id, name) VALUES (?, ?)",
            (category_id, name)
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Observation (per-frame)
    # ------------------------------------------------------------------

    def add_observation(
        self,
        timestamp_ns,
        track_id=None,
        category=None,
        image_id=None,
        image_file_name=None,
        centroid=None,          # (x, y, z)
        bbox3d_min=None,        # (x, y, z)
        bbox3d_max=None,        # (x, y, z)
        point_count=0,
        pcd_path=None,
        mask_path=None,
        bbox_2d=None,           # [x, y, w, h] or None
        tf_translation=None,    # (x, y, z) sensor pose in camera_init frame
        tf_rotation=None,       # (qx, qy, qz, qw) sensor orientation
    ):
        """
        Insert a single per-frame observation row.

        Args:
            timestamp_ns: LiDAR/image timestamp in nanoseconds.
            track_id: COCO track_id (groups observations of same object).
            category: Category name string (e.g. "Poster").
            image_id: COCO image_id.
            image_file_name: Source image filename.
            centroid: (x, y, z) tuple — 3D centroid in camera_init frame.
            bbox3d_min: (x, y, z) tuple — min corner of 3D bounding box.
            bbox3d_max: (x, y, z) tuple — max corner of 3D bounding box.
            point_count: Number of LiDAR points in this observation.
            pcd_path: Path to the per-observation PCD file.
            mask_path: Path to the mask PNG file.
            bbox_2d: [x, y, w, h] list from COCO annotation.
            tf_translation: (x, y, z) sensor translation in camera_init frame.
            tf_rotation: (qx, qy, qz, qw) sensor orientation quaternion.

        Returns:
            The rowid of the inserted observation.
        """
        cx, cy, cz = centroid if centroid else (None, None, None)
        minx, miny, minz = bbox3d_min if bbox3d_min else (None, None, None)
        maxx, maxy, maxz = bbox3d_max if bbox3d_max else (None, None, None)
        bbox_2d_json = json.dumps(bbox_2d) if bbox_2d else None
        tx, ty, tz = tf_translation if tf_translation else (None, None, None)
        qx, qy, qz, qw = tf_rotation if tf_rotation else (None, None, None, None)

        cur = self.conn.execute(
            """
            INSERT INTO observations (
                timestamp_ns, track_id, category, image_id, image_file_name,
                centroid_x, centroid_y, centroid_z,
                bbox3d_min_x, bbox3d_min_y, bbox3d_min_z,
                bbox3d_max_x, bbox3d_max_y, bbox3d_max_z,
                point_count, pcd_path, mask_path, bbox_2d,
                tf_translation_x, tf_translation_y, tf_translation_z,
                tf_rotation_x, tf_rotation_y, tf_rotation_z, tf_rotation_w
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp_ns, track_id, category, image_id, image_file_name,
                cx, cy, cz,
                minx, miny, minz,
                maxx, maxy, maxz,
                point_count, pcd_path, mask_path, bbox_2d_json,
                tx, ty, tz,
                qx, qy, qz, qw
            )
        )
        self.conn.commit()
        return cur.lastrowid

    # ------------------------------------------------------------------
    # Object (aggregated per track_id)
    # ------------------------------------------------------------------

    def upsert_object(
        self,
        track_id,
        category=None,
        centroid=None,
        bbox3d_min=None,
        bbox3d_max=None,
        total_point_count=0,
        observation_count=0,
        first_seen_ns=None,
        last_seen_ns=None,
        aggregated_pcd_path=None,
        tf_translation=None,    # (x, y, z) sensor pose in camera_init frame
        tf_rotation=None,       # (qx, qy, qz, qw) sensor orientation
    ):
        """
        Insert or update the aggregated object row for a given track_id.

        If the track_id already exists, all provided fields are updated.
        If track_id is None, this is a no-op.

        Args:
            track_id: Unique object tracker ID.
            category: Category name.
            centroid: (x, y, z) aggregated centroid.
            bbox3d_min: (x, y, z) overall 3D bbox min corner.
            bbox3d_max: (x, y, z) overall 3D bbox max corner.
            total_point_count: Total points across all observations.
            observation_count: Number of observations for this track.
            first_seen_ns: First timestamp this object was seen.
            last_seen_ns: Last timestamp this object was seen.
            aggregated_pcd_path: Path to the merged per-track PCD.
            tf_translation: (x, y, z) sensor translation in camera_init frame.
            tf_rotation: (qx, qy, qz, qw) sensor orientation quaternion.
        """
        if track_id is None:
            return

        cx, cy, cz = centroid if centroid else (None, None, None)
        minx, miny, minz = bbox3d_min if bbox3d_min else (None, None, None)
        maxx, maxy, maxz = bbox3d_max if bbox3d_max else (None, None, None)
        tx, ty, tz = tf_translation if tf_translation else (None, None, None)
        qx, qy, qz, qw = tf_rotation if tf_rotation else (None, None, None, None)

        self.conn.execute(
            """
            INSERT INTO objects (
                track_id, category,
                centroid_x, centroid_y, centroid_z,
                bbox3d_min_x, bbox3d_min_y, bbox3d_min_z,
                bbox3d_max_x, bbox3d_max_y, bbox3d_max_z,
                total_point_count, observation_count,
                first_seen_ns, last_seen_ns,
                aggregated_pcd_path,
                tf_translation_x, tf_translation_y, tf_translation_z,
                tf_rotation_x, tf_rotation_y, tf_rotation_z, tf_rotation_w
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(track_id) DO UPDATE SET
                category              = excluded.category,
                centroid_x            = excluded.centroid_x,
                centroid_y            = excluded.centroid_y,
                centroid_z            = excluded.centroid_z,
                bbox3d_min_x          = excluded.bbox3d_min_x,
                bbox3d_min_y          = excluded.bbox3d_min_y,
                bbox3d_min_z          = excluded.bbox3d_min_z,
                bbox3d_max_x          = excluded.bbox3d_max_x,
                bbox3d_max_y          = excluded.bbox3d_max_y,
                bbox3d_max_z          = excluded.bbox3d_max_z,
                total_point_count     = excluded.total_point_count,
                observation_count     = excluded.observation_count,
                first_seen_ns         = excluded.first_seen_ns,
                last_seen_ns          = excluded.last_seen_ns,
                aggregated_pcd_path   = excluded.aggregated_pcd_path,
                tf_translation_x      = excluded.tf_translation_x,
                tf_translation_y      = excluded.tf_translation_y,
                tf_translation_z      = excluded.tf_translation_z,
                tf_rotation_x         = excluded.tf_rotation_x,
                tf_rotation_y         = excluded.tf_rotation_y,
                tf_rotation_z         = excluded.tf_rotation_z,
                tf_rotation_w         = excluded.tf_rotation_w
            """,
            (
                track_id, category,
                cx, cy, cz,
                minx, miny, minz,
                maxx, maxy, maxz,
                total_point_count, observation_count,
                first_seen_ns, last_seen_ns,
                aggregated_pcd_path,
                tx, ty, tz,
                qx, qy, qz, qw
            )
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all_objects(self):
        """Return all rows from the objects table as list of dicts."""
        cur = self.conn.execute("SELECT * FROM objects ORDER BY track_id")
        return [dict(row) for row in cur.fetchall()]

    def get_observations(self, track_id):
        """Return all observations for a given track_id."""
        cur = self.conn.execute(
            "SELECT * FROM observations WHERE track_id = ? ORDER BY timestamp_ns",
            (track_id,)
        )
        return [dict(row) for row in cur.fetchall()]

    def get_observation_count(self, track_id):
        """Return the number of observations for a track_id."""
        cur = self.conn.execute(
            "SELECT COUNT(*) FROM observations WHERE track_id = ?",
            (track_id,)
        )
        return cur.fetchone()[0]

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()