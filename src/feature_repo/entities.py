from feast import Entity

# Define an entity for the user
user = Entity(
    name="user",  # Name of the entity
    join_keys=["user_id"],  # Field name used to join this entity with feature views
    description="A user in the system",
)

# Define an entity for the post
post = Entity(
    name="post",  # Name of the entity
    join_keys=["post_id"],  # Field name used to join this entity with feature views
    description="A post created by a user",
)

# Note: In older versions of Feast, `join_key` (singular) was used.
# In recent versions (like 0.30+), `join_keys` (plural, as a list) is preferred,
# even for single join keys, for consistency with composite keys.
# The task description uses `join key` (singular), but I'll use `join_keys`
# as it's the more current and flexible approach.