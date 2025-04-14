from typing import List, Dict, Optional
from contextlib import closing


# import sqlite3
# conn = sqlite3.connect("movies.db")
import psycopg2
conn = psycopg2.connect("dbname=postgres user=u1319464 port=2111")

import numpy as np
import requests
import json
import cv2
import os

# # Define a custom adapter and typecaster for the vector type
# from psycopg2.extensions import register_adapter, register_type, new_type
# from psycopg2.extras import CompositeCaster


# class VectorCaster(CompositeCaster):
#     def make(self, values):
#         print(values)
#         return np.array(json.loads(values), dtype=np.float32)
#         # return np.array(values, dtype=np.float32)

# def register_vector():
#     VECOID = 33000  # This is the OID for the vector type in your PostgreSQL database
#     VECTOR = new_type((VECOID,), 'VECTOR', VectorCaster)
#     register_type(VECTOR)

# # Register the vector type
# register_vector()
import pgvector.psycopg2
pgvector.psycopg2.register_vector(conn) #register pgvector's types


class OMDBApiCache:
    cache = {}
    cache_dir = "omdb"
    
    def __init__(self, api_key: str):
        self.base_url = "http://www.omdbapi.com/"
        self.api_key = api_key
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
    
    def get_movie_data(self, identifier: str, is_imdb_id: bool = False):
        if identifier in OMDBApiCache.cache:
            return OMDBApiCache.cache[identifier]
        
        params = {
            'apikey': self.api_key,
            'i' if is_imdb_id else 't': identifier
        }
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200:
            data = response.json()
            if data.get("Response") == "True":
                imdb_id = data.get("imdbID")
                if imdb_id:
                    file_path = os.path.join(self.cache_dir, f"{imdb_id}.json")
                    with open(file_path, "w") as f:
                        json.dump(data, f)
                else:
                    raise ValueError("No IMDb ID found in response")
                OMDBApiCache.cache[identifier] = data
            else:
                OMDBApiCache.cache[identifier] = None
        else:
            OMDBApiCache.cache[identifier] = None
        
        return OMDBApiCache.cache[identifier]
    
    def get_movie_ratings(self, identifier: str, is_imdb_id: bool = False):
        data = self.get_movie_data(identifier, is_imdb_id)
        if data and "Ratings" in data:
            return data["Ratings"]
        return None


omdb = OMDBApiCache(api_key='15b0ce0f') # not really a secret, can be generated for free at https://www.omdbapi.com/


class Movie():

    @staticmethod
    def add_movie_from_omdb_data(file_path, omdb_data):
        m = Movie(imdb_id=omdb_data.get("imdbID"))
        
        with closing(conn.cursor()) as cursor:
            
            cursor.execute("""
                INSERT INTO movie (file_path, title, date_created, imdb_id)
                VALUES (%s, %s, %s, %s)
            """, (file_path, omdb_data.get("Title"), omdb_data.get("Released"), omdb_data.get("imdbID")))
            
            conn.commit()
            return m

    @staticmethod
    def iterate_all_movies():
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT imdb_id FROM movie")
            return [ Movie(res[0]) for res in cursor.fetchall()]

    def __init__(self, imdb_id):
        super().__init__()
        self.imdb_id = imdb_id

    def get_file_path(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT file_path FROM movie WHERE imdb_id = %s", (self.get_imdb_id(),))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_title(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT title FROM movie WHERE imdb_id = %s", (self.get_imdb_id(),))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_imdb_id(self):
        return self.imdb_id

    def get_date_created(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT date_created FROM movie WHERE imdb_id = %s", (self.get_imdb_id(),))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_omdb_data(self):
        return omdb.get_movie_data(identifier=self.get_imdb_id(), is_imdb_id=True)

    def get_file_handle(self):
        file_path = self.get_file_path()
        if file_path and os.path.exists(file_path):
            return cv2.VideoCapture(file_path)
        return None

    def iterate_frames(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("""
                SELECT frame_index FROM frame
                WHERE imdb_id = %s
            """, (self.get_imdb_id(),))
            return [ Frame(self.imdb_id, res[0]) for res in cursor.fetchall() ]
    # def iterate_frames(self):
    #     cursor = conn.cursor()
    #     cursor.execute("""
    #         SELECT imdb_id, frame_index FROM frame
    #         WHERE imdb_id = %s
    #     """, (self.get_imdb_id(),))
        
    #     for res in cursor:
    #         yield Frame(res[0], res[1])

    def __str__(self):
        return self.get_title()

import imagehash
import hashlib
from PIL import Image

class Frame():
    movie_cache = {}

    @staticmethod
    def add_frame(imdb_id, frame_index, frame_image=None):
        f = Frame(imdb_id, frame_index, frame_image=frame_image)
        with closing(conn.cursor()) as cursor:
    
            cursor.execute("""
                INSERT INTO frame (imdb_id, frame_index)
                VALUES (%s, %s)
                ON CONFLICT (imdb_id, frame_index) DO NOTHING;
            """, (imdb_id, frame_index))
        
        conn.commit()
        return f

    def __init__(self, imdb_id, frame_index, frame_image=None):
        super().__init__()
        self.imdb_id = imdb_id
        self.frame_index = frame_index
        self.frame_image = frame_image

    @staticmethod
    def get_movie(imdb_id):
        if imdb_id not in Frame.movie_cache:
            Frame.movie_cache[imdb_id] = Movie(imdb_id)
        return Frame.movie_cache[imdb_id]

    # def get_frame_timestamp(self):
    #     movie = Frame.get_movie(self.imdb_id)
    #     video = movie.get_file_handle()
    #     if not video:
    #         return None
        
    #     fps = video.get(cv2.CAP_PROP_FPS)
    #     return self.frame_index / fps if fps and fps > 0 else None

    def get_frame_image(self):
        if not self.frame_image is None:
            return self.frame_image
        
        movie = Frame.get_movie(self.imdb_id)
        video = movie.get_file_handle()
        if not video:
            raise Exception('Could not find image frame; video is not specificed properly.')
        
        video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        success, frame = video.read()
        self.frame_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if success else None
        return self.frame_image

    def get_cached_hash(self, hash_type):
        with closing(conn.cursor()) as cursor:
            cursor.execute(f"SELECT {hash_type} FROM frame WHERE imdb_id = %s AND frame_index = %s", (self.imdb_id, self.frame_index))
            result = cursor.fetchone()
            return result[0] if result and result[0] else None

    def compute_and_cache_hash(self, hash_func, hash_type):
        frame_image = self.get_frame_image()
        
        hash_value = hash_func(frame_image)
        with closing(conn.cursor()) as cursor:
            cursor.execute(f"""
                UPDATE frame SET {hash_type} = %s
                WHERE imdb_id = %s AND frame_index = %s
            """, (hash_value, self.imdb_id, self.frame_index))
        conn.commit()
        return hash_value

    def get_wavelet_hash(self):
        whash = lambda x: str(imagehash.whash(x))
        return self.get_cached_hash("wavelet_hash") or self.compute_and_cache_hash(whash, "wavelet_hash")

    def get_a_hash(self):
        average_hash = lambda x: str(imagehash.average_hash(x))
        return self.get_cached_hash("a_hash") or self.compute_and_cache_hash(average_hash, "a_hash")

    def get_d_hash(self):
        dhash = lambda x: str(imagehash.dhash(x))
        return self.get_cached_hash("d_hash") or self.compute_and_cache_hash(dhash, "d_hash")

    def get_perceptual_hash(self):
        phash = lambda x: str(imagehash.phash(x))
        return self.get_cached_hash("perceptual_hash") or self.compute_and_cache_hash(phash, "perceptual_hash")

    def get_md5_hash(self):
        compute_md5_on_img = lambda img: str(hashlib.md5(img.tobytes()).hexdigest())
        return self.get_cached_hash("md5_hash") or self.compute_and_cache_hash(compute_md5_on_img, "md5_hash")
        
        # cached_md5 = self.get_cached_hash("md5_hash")
        # if cached_md5:
        #     return cached_md5
        
        # if frame_image is None:
        #     return None
        
        # img_bytes = img.tobytes()
        # md5_hash = hashlib.md5(img_bytes).hexdigest()
        
        # with closing(conn.cursor()) as cursor:
        # cursor.execute("""
        #     UPDATE frame SET md5_hash = %s
        #     WHERE imdb_id = %s AND frame_index = %s
        # """, (md5_hash, self.imdb_id, self.frame_index))
        # self.get_connection().commit()
        
        # return md5_hash

    def get_average_color(self):
        compute_mean_color = lambda img: np.mean(np.array(img), axis=(0, 1))
        # ret = self.get_cached_hash("average_color") or self.compute_and_cache_hash(compute_mean_color, "average_color")

        with closing(conn.cursor()) as cursor:
            cursor.execute(f"SELECT average_color FROM frame WHERE imdb_id = %s AND frame_index = %s", (self.imdb_id, self.frame_index))
            result = cursor.fetchone()
            if not result[0] is None:
                return result[0]

            frame_image = self.get_frame_image()

            average_color = compute_mean_color(frame_image)

            cursor.execute("""
                UPDATE frame
                SET average_color = %s
                WHERE imdb_id = %s AND frame_index = %s
            """, (average_color.tolist(), self.imdb_id, self.frame_index))
            
        return average_color

    def get_num_faces(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute(f"SELECT num_faces FROM frame WHERE imdb_id = %s AND frame_index = %s", (self.imdb_id, self.frame_index))
            result = cursor.fetchone()
        if result and result[0]:
            return result[0]
        else:
            raise Exception('Frame not fully calculated: unknown number of faces');

    def set_num_faces(self, num_faces):
        with closing(conn.cursor()) as cursor:
            cursor.execute(f"""
                UPDATE frame SET num_faces = %s
                WHERE imdb_id = %s AND frame_index = %s
            """, (num_faces, self.imdb_id, self.frame_index))
        conn.commit()

    def compute_frame(self):
        self.get_wavelet_hash()
        self.get_a_hash()
        self.get_d_hash()
        self.get_perceptual_hash()
        self.get_md5_hash()
        self.get_average_color()

    def is_fully_cached(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("""
                SELECT * FROM frame
                WHERE frame_index = %s
            """, (self.frame_index,)
            )

            result = cursor.fetchone()
        
        if result and not any(elem is None for elem in result):
            return True
        else:
            return False

    def iterate_faces(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("""
                SELECT face_index FROM face
                WHERE imdb_id = %s AND frame_index = %s
            """, (self.imdb_id, self.frame_index))
            return [ Face(self.imdb_id, self.frame_index, res[0]) for res in cursor.fetchall() ]


class Actor():

    @staticmethod
    def add_actor(full_name: str):
        a = Actor(full_name)
        with closing(conn.cursor()) as cursor:
            
            # Insert actor data into table
            cursor.execute("""
                INSERT INTO actor (full_name)
                VALUES (%s)
            """, (full_name,))
        
        conn.commit()

    def __init__(self, full_name: str):
        super().__init__()
        self.full_name = full_name

    def get_actor_id(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT actor_id FROM actor WHERE full_name = %s", (self.full_name,))
            result = cursor.fetchone()
            return result[0] if result else None

    def get_name(self):
        return self.full_name


class Character():

    @staticmethod
    def add_character():
        with closing(conn.cursor()) as cursor:
            
            # Insert character data into table
            cursor.execute("""
                INSERT INTO character (actor_id, full_name) VALUES (NULL, NULL)
            """)

        character_id = cur.fetchone()[0]
        
        conn.commit()

        c = Character(character_id)

    def __init__(self, character_id):
        super().__init__()
        self.character_id = character_id

    def get_character_id(self):
        # with closing(conn.cursor()) as cursor:
        #     cursor.execute("SELECT character_id FROM character WHERE full_name = %s", (self.full_name,))
        #     result = cursor.fetchone()
        # return result[0] if result else None
        return self.character_id

    # def get_name(self):
    #     return self.full_name

    # def get_movie_id(self):
    #     with closing(conn.cursor()) as cursor:
    #         cursor.execute("SELECT movie_id FROM movie_character WHERE character_id = %s", (self.get_character_id(),))
    #         result = cursor.fetchone()
    #     return result[0] if result else None

    # def get_actor_id(self):
    #     return self.actor_id


class ImageCache:
    def __init__(self, cache_dir: str):
        # Initialize the cache directory
        self.cache_dir = cache_dir
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)

    def add_image(self, pil_image: Image.Image, image_id: str):
        """Add a PIL image to the cache by its ID."""
        # Generate the target file path for the image
        target_path = os.path.join(self.cache_dir, f"{image_id}.png")

        try:
            # Save the PIL image to the cache directory
            pil_image.save(target_path, format='PNG')
        except Exception as e:
            print(f"Error adding image {image_id}.png: {e}")

    def get_image(self, image_id: str):
        """Retrieve an image from the cache by its ID."""
        # Generate the path for the image in the cache
        image_path = os.path.join(self.cache_dir, f"{image_id}.png")

        if os.path.exists(image_path):
            return Image.open(image_path)
        else:
            return None

    def remove_image(self, image_id: str):
        """Remove an image from the cache."""
        image_path = os.path.join(self.cache_dir, f"{image_id}.png")

        if os.path.exists(image_path):
            os.remove(image_path)
            print(f"Image {image_id}.png removed from cache.")
        else:
            print(f"Image {image_id}.png not found in cache.")

face_image_cache = ImageCache('images/face')

# from facelib import FaceDetector, EmotionDetector
# face_detector = FaceDetector()
# emotion_detector = EmotionDetector()

# from facelib.RetinaFace.utils.alignment import get_reference_facial_points

# similarity_transform = transform.SimilarityTransform()
# def align_image():
#     similarity_transform
    

# how to get bounding boxes:
# faces, boxes, scores, landmarks = face_detector.detect_align(frame)

# # Recognize facial expression (happy, sad, angry, etc.)
# emotions, probab = emotion_detector.detect_emotion(faces)

# initially, every face may not have a know character so don't try to classify faces upon adding a face

class Face():

    @staticmethod
    def add_face(imdb_id: str, frame_index: int, face_index: int):
        f = Face(imdb_id, frame_index, face_index)
        with closing(conn.cursor()) as cursor:
            # face_embedding BLOB,
            
            # Insert face data into table
            cursor.execute("""
                INSERT INTO face (imdb_id, frame_index, face_index)
                VALUES (%s, %s, %s)
                ON CONFLICT (imdb_id, frame_index, face_index) DO NOTHING;
            """, (imdb_id, frame_index, face_index,))
        
            conn.commit()
        return f

    def __init__(self, imdb_id: str, frame_index: int, face_index: int):
        super().__init__()
        self.imdb_id = imdb_id
        self.frame_index = frame_index
        self.face_index = face_index
        self.face_image_aligned = None

    def get_face_index(self):
        return self.face_index

    def get_frame_index(self):
        return self.frame_index

    def get_character_id(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT character_id FROM face WHERE imdb_id = %s AND frame_index = %s AND face_index = %s",
                           (self.imdb_id, self.frame_index, self.face_index))
            result = cursor.fetchone()
        if result:
            return result[0]
        else:
            raise Exception("No character defined for movie/frame/index")

    def set_face_image_aligned(self, face_image_aligned) -> Image.Image:

        self.face_image_aligned = face_image_aligned
        face_image_cache_key = f'{self.imdb_id}_{self.frame_index:09d}_{self.face_index:03d}'
        face_image_cache.add_image(face_image_aligned, face_image_cache_key)


    def get_face_image_aligned(self) -> Image.Image:

        # memory cache
        if self.face_image_aligned:
            return self.face_image_aligned

        # disk cache
        face_image_cache_key = f'{self.imdb_id}_{self.frame_index:09d}_{self.face_index:03d}'
        
        face_image = face_image_cache.get_image(face_image_cache_key)
        if (face_image):
            self.face_image_aligned = face_image
            return face_image

        raise Exception("Face not aligned yet")
        
        # movie = Frame.get_movie(self.imdb_id)
        # video = movie.get_file_handle()
        # if not video:
        #     raise Exception(f"No Video found for imdb_id: {self.imdb_id}")
        
        # video.set(cv2.CAP_PROP_POS_FRAMES, self.frame_index)
        # success, frame = video.read()
        # if not success:
        #     raise ErExceptionror(f"No frame found for imdb_id, frame_index: {self.imdb_id}, {self.frame_index}")
        
        # facial_landmarks = self.get_facial_landmarks()
        # face = frame[y:y+h, x:x+w]
        # face_image = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        
        # face_image_cache.add_image(face_image_cache_key, Image.fromarray(face_image))

        # return face_image_cache.get_image(face_image_cache_key)

    def get_facial_landmarks(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT facial_landmarks FROM face WHERE imdb_id = %s AND frame_index = %s AND face_index = %s",
                           (self.imdb_id, self.frame_index, self.face_index))
            result = cursor.fetchone()
        if result[0]:
            return np.array(result[0])
        else:
            raise Exception("No facial_landmarks defined for movie/frame/index")

    def set_facial_landmarks(self, vector: np.ndarray):
        """Insert a vector into the database, associated with the given key"""
        with closing(conn.cursor()) as cursor:
            cursor.execute('''
                INSERT INTO face (imdb_id, frame_index, face_index, facial_landmarks)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (imdb_id, frame_index, face_index)
                DO UPDATE SET facial_landmarks = EXCLUDED.facial_landmarks
            ''', (self.imdb_id, self.frame_index, self.face_index, vector.ravel().tolist()))
            conn.commit()

    def get_cached_hash(self, hash_type):
        with closing(conn.cursor()) as cursor:
            cursor.execute(f"SELECT {hash_type} FROM face WHERE imdb_id = %s AND frame_index = %s AND face_index = %s", (self.imdb_id, self.frame_index, self.face_index))
            result = cursor.fetchone()
        return result[0] if result and result[0] else None

    def compute_and_cache_hash(self, hash_func, hash_type):
        frame_image = self.get_face_image_aligned()
        if not frame_image:
            raise Exception("No image to compute hash on")
        
        hash_value = str(hash_func(frame_image))
        with closing(conn.cursor()) as cursor:
            cursor.execute(f"""
                UPDATE face SET {hash_type} = %s
                WHERE imdb_id = %s AND frame_index = %s AND face_index = %s
            """, (hash_value, self.imdb_id, self.frame_index, self.face_index))
        conn.commit()
        return hash_value

    def get_wavelet_hash(self):
        return self.get_cached_hash("wavelet_hash") or self.compute_and_cache_hash(imagehash.whash, "wavelet_hash")

    def get_a_hash(self):  # average hashing
        return self.get_cached_hash("a_hash") or self.compute_and_cache_hash(imagehash.average_hash, "a_hash")

    def get_d_hash(self):  # difference hashing
        return self.get_cached_hash("d_hash") or self.compute_and_cache_hash(imagehash.dhash, "d_hash")

    def get_perceptual_hash(self):
        return self.get_cached_hash("perceptual_hash") or self.compute_and_cache_hash(imagehash.phash, "perceptual_hash")

    def get_md5_hash(self):
        cached_md5 = self.get_cached_hash("md5_hash")
        if cached_md5:
            return cached_md5
        
        face_image = self.get_face_image_aligned()
        if not face_image:
            return None
        
        img_bytes = face_image.tobytes()
        md5_hash = hashlib.md5(img_bytes).hexdigest()
        
        with closing(conn.cursor()) as cursor:
            cursor.execute("""
                UPDATE face SET md5_hash = %s
                WHERE imdb_id = %s AND frame_index = %s AND face_index = %s
            """, (md5_hash, self.imdb_id, self.frame_index, self.face_index))
        conn.commit()
        
        return md5_hash

    def compute_face(self):
        self.get_wavelet_hash()
        self.get_a_hash()
        self.get_d_hash()
        self.get_perceptual_hash()
        self.get_md5_hash()

    def get_emotion_embeddings(self):

        with closing(conn.cursor()) as cursor:
            cursor.execute("""
                SELECT emb_method FROM face_emotion
                WHERE imdb_id = %s AND frame_index = %s
            """, (self.imdb_id, self.frame_index))
            return [ FaceEmotionEmbedding(self.imdb_id, self.frame_index, self.face_index, res[0]) for res in cursor.fetchall() ]

    def get_face_embeddings(self):

        with closing(conn.cursor()) as cursor:
            cursor.execute("""
                SELECT emb_method FROM face_emotion
                WHERE imdb_id = %s AND frame_index = %s
            """, (self.imdb_id, self.frame_index))
            return [ FaceIdentity(self.imdb_id, self.frame_index, self.face_index, res[0]) for res in cursor.fetchall() ]



class FaceEmotionEmbedding():

    @staticmethod
    def add_face_emotion(imdb_id: str, frame_index: int, face_index: int, emb_method: str):
        f = FaceEmotionEmbedding(imdb_id, frame_index, face_index, emb_method)
        with closing(conn.cursor()) as cursor:
        
            # Insert face data into table
            cursor.execute("""
                INSERT INTO face_emotion (imdb_id, frame_index, face_index, emb_method)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (imdb_id, frame_index, face_index, emb_method) DO NOTHING;
            """, (imdb_id, frame_index, face_index, emb_method))
        
            conn.commit()
        return f

    def __init__(self, imdb_id: str, frame_index: int, face_index: int, emb_method: str):
        super().__init__()
        self.imdb_id = imdb_id
        self.frame_index = frame_index
        self.face_index = face_index
        self.emb_method = emb_method

    def get_face_index(self):
        return self.face_index

    def get_frame_index(self):
        return self.frame_index

    def get_embedding_method(self):
        return self.emb_method

    def get_embedding(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT embedding FROM face_emotion WHERE imdb_id = %s AND frame_index = %s AND face_index = %s",
                           (self.imdb_id, self.frame_index, self.face_index))
            result = cursor.fetchone()
        if result:
            return result[0]
        else:
            raise Exception("No emotion_embedding defined for movie/frame/index")

    def get_classification_confidence(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT classification, confidence FROM face_emotion WHERE imdb_id = %s AND frame_index = %s AND face_index = %s",
                           (self.imdb_id, self.frame_index, self.face_index))
            result = cursor.fetchone()
        if result:
            return result[0], result[1]
        else:
            raise Exception("No emotion_embedding defined for movie/frame/index")

    def set_embedding(self, vector: np.ndarray, classification: str, confidence: float):
        """Insert a vector into the database, associated with the given key"""
        with closing(conn.cursor()) as cursor:
            # cursor.execute('''
            #     INSERT INTO face_embeddings (imdb_id, frame_index, face_index, embedding, classification, confidence) VALUES (%s, %s, %s, %s, %s, %s)
            # ''', (self.imdb_id, self.frame_index, self.face_index, vector, classification, confidence))

            # cursor.execute("""
            #     UPDATE face_embeddings
            #     SET average_color = %s
            #     WHERE imdb_id = %s AND frame_index = %s
            # """, (average_color.tolist(), self.imdb_id, self.frame_index))

            # cursor.execute('''
            #     INSERT INTO face_embeddings (imdb_id, frame_index, face_index, embedding, classification, confidence) 
            #     VALUES (%s, %s, %s, %s, %s, %s)
            # ''', (self_imdb_id, self_frame_index, self_face_index, vector_str, classification, confidence))

            cursor.execute('''
                UPDATE face_emotion 
                SET embedding = %s, classification = %s, confidence = %s
                WHERE imdb_id = %s AND frame_index = %s AND face_index = %s
            ''', (vector, classification, confidence, self.imdb_id, self.frame_index, self.face_index))


        conn.commit()



class FaceIdentity():

    @staticmethod
    def add_face(imdb_id: str, frame_index: int, face_index: int, emb_method: str):
        f = FaceIdentity(imdb_id, frame_index, face_index, emb_method)
        with closing(conn.cursor()) as cursor:
        
            # Insert face data into table
            cursor.execute("""
                INSERT INTO face_identity (imdb_id, frame_index, face_index, emb_method)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (imdb_id, frame_index, face_index, emb_method) DO NOTHING;
            """, (imdb_id, frame_index, face_index, emb_method))
        
            conn.commit()
        return f

    def __init__(self, imdb_id: str, frame_index: int, face_index: int, emb_method: str):
        super().__init__()
        self.imdb_id = imdb_id
        self.frame_index = frame_index
        self.face_index = face_index
        self.emb_method = emb_method

    def get_face_index(self):
        return self.face_index

    def get_frame_index(self):
        return self.frame_index

    def get_embedding_method(self):
        return self.emb_method

    def get_embedding(self):
        with closing(conn.cursor()) as cursor:
            cursor.execute("SELECT embedding FROM face_identity WHERE imdb_id = %s AND frame_index = %s AND face_index = %s",
                           (self.imdb_id, self.frame_index, self.face_index))
            result = cursor.fetchone()
        if result:
            return result[0]
        else:
            raise Exception("No emotion_embedding defined for movie/frame/index in face_identity")
    
    def set_embedding(self, vector: np.ndarray):
        """Insert a vector into the database, associated with the given key"""
        with closing(conn.cursor()) as cursor:
            
            cursor.execute('''
                UPDATE face_identity 
                SET embedding = %s
                WHERE imdb_id = %s AND frame_index = %s AND face_index = %s
            ''', (vector, self.imdb_id, self.frame_index, self.face_index))

        conn.commit()

    def set_character(self, character_id):
        with closing(conn.cursor()) as cursor:
            
            cursor.execute('''
                UPDATE face_identity 
                SET character_id = %s
                WHERE imdb_id = %s AND frame_index = %s AND face_index = %s
            ''', (character_id, self.imdb_id, self.frame_index, self.face_index))

        conn.commit()

    def get_character(self):
        with closing(conn.cursor()) as cursor:
            
            cursor.execute('''
                SELECT character_id FROM face_identity
                WHERE imdb_id = %s AND frame_index = %s AND face_index = %s
            ''', (self.imdb_id, self.frame_index, self.face_index))
            result = cursor.fetchone()

        if result[0]:
            return new Character(result[0])
        else:
            raise Exception("No character_id defined for movie/frame/index in face_identity")