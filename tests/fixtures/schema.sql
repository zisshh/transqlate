CREATE TABLE users (
    id INTEGER PRIMARY KEY,
    name TEXT
);
CREATE TABLE posts (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    content TEXT,
    FOREIGN KEY(user_id) REFERENCES users(id)
);
