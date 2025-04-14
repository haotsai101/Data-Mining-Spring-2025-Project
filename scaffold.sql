
CREATE EXTENSION VECTOR;

CREATE TABLE IF NOT EXISTS movie (
    file_path TEXT,
    title TEXT,
    imdb_id TEXT,
    date_created TIMESTAMP,
    PRIMARY KEY (imdb_id)
);


CREATE TABLE IF NOT EXISTS frame (
    imdb_id TEXT,
    frame_index INTEGER,
    wavelet_hash TEXT,
    a_hash TEXT,
    d_hash TEXT,
    perceptual_hash TEXT,
    md5_hash TEXT,
    average_color VECTOR(3),
    num_faces INTEGER,
    PRIMARY KEY (imdb_id, frame_index)
);


CREATE TABLE IF NOT EXISTS actor (
    actor_id SERIAL PRIMARY KEY,
    full_name TEXT
);


CREATE TABLE IF NOT EXISTS character (
    character_id SERIAL PRIMARY KEY,
    actor_id INTEGER,
    full_name TEXT,
    FOREIGN KEY (actor_id) REFERENCES actor (actor_id)
);


CREATE TABLE IF NOT EXISTS face (
    imdb_id TEXT,
    frame_index INTEGER,
    face_index INTEGER,
    wavelet_hash TEXT,
    a_hash TEXT,
    d_hash TEXT,
    perceptual_hash TEXT,
    md5_hash TEXT,
    character_id INTEGER,
    facial_landmarks VECTOR(10),
    PRIMARY KEY (imdb_id, frame_index, face_index),
    FOREIGN KEY (character_id) REFERENCES character (character_id)
);

    
CREATE TABLE IF NOT EXISTS face_emotion (
    imdb_id TEXT,
    frame_index INTEGER,
    face_index INTEGER,
    emb_method TEXT,
    embedding VECTOR(7),
    classification TEXT,
    confidence REAL,
    PRIMARY KEY (imdb_id, frame_index, face_index, emb_method)
);

CREATE TABLE IF NOT EXISTS face_identity (
    imdb_id TEXT,
    frame_index INTEGER,
    face_index INTEGER,
    emb_method TEXT,
    embedding VECTOR(512),
    character_id INTEGER,
    PRIMARY KEY (imdb_id, frame_index, face_index, emb_method),
    FOREIGN KEY (character_id) REFERENCES character (character_id)
);