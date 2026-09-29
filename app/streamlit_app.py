import os

import requests
import streamlit as st


API_URL = os.getenv("RECOMMENDER_API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Movie Finder", page_icon="🎬", layout="wide")
st.title("🎬 Movie Finder")
st.caption("Discover films selected for your taste, explore similar titles, and review your history.")

with st.sidebar:
    st.header("Find movies")
    user_id = st.number_input("MovieLens user ID", min_value=1, max_value=100000, value=1)
    count = st.slider("Number of results", min_value=1, max_value=30, value=10)
    st.caption(f"API: {API_URL}")


def api_get(path: str, params: dict | None = None) -> dict | None:
    try:
        response = requests.get(f"{API_URL}{path}", params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as error:
        detail = error.response.text if error.response is not None else str(error)
        st.error(f"Could not reach the recommendation API: {detail}")
        return None


tabs = st.tabs(["For you", "Similar movies", "Your profile", "History"])
with tabs[0]:
    if st.button("Get recommendations", type="primary"):
        data = api_get(f"/recommend/{int(user_id)}", {"limit": count})
        if data:
            if not data["recommendations"]:
                st.info("No recommendations are available for this user yet.")
            for movie in data["recommendations"]:
                st.markdown(f"**{movie['title']}** · {movie['genres']}  \nMatch score: {movie['score']:.3f}")

with tabs[1]:
    movie_id = st.number_input("Movie ID", min_value=1, max_value=100000, value=1)
    if st.button("Find similar movies"):
        data = api_get(f"/similar/{int(movie_id)}", {"limit": count})
        if data:
            for movie in data["similar_items"]:
                st.markdown(f"**{movie['title']}** · {movie['genres']}  \nSimilarity: {movie['score']:.3f}")

with tabs[2]:
    if st.button("Load profile"):
        data = api_get(f"/profile/{int(user_id)}")
        if data:
            left, middle, right = st.columns(3)
            left.metric("Ratings", data["rating_count"])
            middle.metric("Average rating", data["average_rating"] or "—")
            right.metric("Activity", data["activity_level"].title())
            if data["favorite_genres"]:
                st.write("Favorite genres:", ", ".join(data["favorite_genres"]))
            if data.get("occupation"):
                st.write("Occupation:", data["occupation"])

with tabs[3]:
    if st.button("Load recommendation history"):
        st.session_state["history_data"] = api_get(
            f"/history/{int(user_id)}", {"limit": 20}
        )
    history_data = st.session_state.get("history_data")
    if history_data:
        if not history_data["entries"]:
            st.info("There is no saved recommendation history for this user.")
        for entry in history_data["entries"]:
            with st.expander(entry["created_at"]):
                for movie in entry["recommendations"]:
                    st.write(f"{movie['title']} · {movie['genres']}")
        if history_data["entries"] and st.button("Clear history for this user"):
            try:
                response = requests.delete(
                    f"{API_URL}/history/{int(user_id)}", timeout=10
                )
                response.raise_for_status()
                st.session_state["history_data"] = {
                    "entries": [],
                    "user_id": int(user_id),
                }
                st.success(f"Deleted {response.json()['deleted_entries']} history entries.")
            except requests.RequestException as error:
                st.error(f"Could not clear history: {error}")
