import os
import requests
import streamlit as st

st.set_page_config(page_title='CineMatch', page_icon='🎞️', layout='wide', initial_sidebar_state='expanded')
if 'CINEMATCH_API_URL' in st.secrets:
    API = st.secrets['CINEMATCH_API_URL'].rstrip('/')
else:
    API = os.getenv('CINEMATCH_API_URL', 'http://localhost:8000').rstrip('/')

st.markdown('''
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1,h2,h3 { font-family: 'Space Grotesk', sans-serif; letter-spacing: -.03em; }
.stApp { background: #0b0d12; color: #f4f1ea; }
[data-testid="stSidebar"] { background: #11141b; border-right: 1px solid #242936; }
.brand { font-family:'Space Grotesk'; font-size: 2rem; font-weight:700; color:#f7c873; margin: .4rem 0 2rem; }
.eyebrow { color:#f7c873; text-transform:uppercase; letter-spacing:.16em; font-size:.72rem; font-weight:700; }
.hero { background:linear-gradient(110deg,#171a24,#202331); border:1px solid #303545; border-radius:20px; padding:2.4rem; margin-bottom:1.5rem; }
.hero h1 { font-size:3.1rem; margin:.35rem 0; }
.muted { color:#a6acba; }
.card { background:#151821; border:1px solid #292e3b; border-radius:14px; padding:1rem 1.1rem; min-height:125px; margin-bottom:.7rem; }
.card h3 { margin:.1rem 0 .35rem; font-size:1.05rem; }
.tag { display:inline-block; color:#b9c0d0; background:#252a37; border-radius:99px; padding:.2rem .55rem; margin:.1rem .2rem .1rem 0; font-size:.72rem; }
.score { color:#f7c873; font-weight:700; float:right; }
.section { margin:1.5rem 0 .7rem; }
div.stButton > button { border-radius:8px; border:1px solid #3a4152; background:#1d2230; color:#f4f1ea; }
div.stButton > button:hover { border-color:#f7c873; color:#f7c873; }
div[data-testid="stMetric"] { background:#151821; border:1px solid #292e3b; padding:1rem; border-radius:12px; }
</style>
''', unsafe_allow_html=True)


def api(method, path, **kwargs):
    try:
        response = requests.request(method, f'{API}{path}', timeout=20, **kwargs)
        if response.ok:
            return response.json()
        st.error(response.json().get('detail', 'Something went wrong.'))
    except requests.RequestException:
        st.error(f'Cannot reach the CineMatch API at {API}. Start the backend or set CINEMATCH_API_URL.')
    return None


def tags(genres):
    return ''.join(f'<span class="tag">{g}</span>' for g in str(genres).split('|') if g and g != '(no genres listed)')


@st.cache_data(ttl=3600, show_spinner=False)
def load_poster(url):
    """Download poster bytes so Streamlit does not rely on browser URL loading."""
    if not url or url == 'N/A':
        return None
    clean_url = str(url).replace('*', '').replace('\\_', '_').strip()
    try:
        response = requests.get(clean_url, headers={'User-Agent': 'CineMatch/1.0'}, timeout=10)
        if response.ok and response.headers.get('content-type', '').startswith('image/'):
            return response.content
    except requests.RequestException:
        pass
    return None


def movie_card(item, account_id=None, action='watch'):
    movie_id = item.get('movie_id', item.get('movieId'))
    score = item.get('predicted_rating')
    score_html = f'<span class="score">★ {score}</span>' if score else ''
    poster = load_poster(item.get('poster', ''))
    if poster:
        st.image(poster, width=150)
    else:
        st.markdown('<div style="height:150px;width:100px;background:#252a37;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#a6acba;font-size:.75rem;text-align:center;">No poster</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="card"><h3>{score_html}{item["title"]}</h3><div>{tags(item.get("genres", ""))}</div></div>', unsafe_allow_html=True)
    if account_id and action == 'watch':
        if st.button('＋ Watchlist', key=f'watch-{movie_id}'):
            api('POST', '/watchlist', json={'account_id': account_id, 'movie_id': int(movie_id)})
            st.toast('Added to watchlist')
    elif account_id and action == 'remove':
        if st.button('Remove', key=f'remove-{movie_id}'):
            api('DELETE', f'/watchlist/{account_id}/{movie_id}')
            st.rerun()


def auth_screen():
    st.markdown('<div class="brand">CineMatch</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero"><div class="eyebrow">Your next favorite film</div><h1>Less scrolling.<br>More watching.</h1><p class="muted">A personal movie space powered by your taste.</p></div>', unsafe_allow_html=True)
    login, signup = st.tabs(['Sign in', 'Create account'])
    with login:
        with st.form('login'):
            email = st.text_input('Email')
            password = st.text_input('Password', type='password')
            if st.form_submit_button('Sign in', type='primary'):
                result = api('POST', '/auth/login', json={'email': email, 'password': password})
                if result:
                    st.session_state.account = result['account']; st.rerun()
    with signup:
        with st.form('signup'):
            name = st.text_input('Name')
            email = st.text_input('Email', key='signup-email')
            password = st.text_input('Password (6+ characters)', type='password', key='signup-password')
            if st.form_submit_button('Create account', type='primary'):
                result = api('POST', '/auth/signup', json={'display_name': name, 'email': email, 'password': password})
                if result:
                    st.session_state.account = result['account']; st.rerun()


def home(account_id):
    st.markdown('<div class="hero"><div class="eyebrow">Personalised for you</div><h1>Find something<br>worth your evening.</h1><p class="muted">Rate a few films and CineMatch will learn your rhythm.</p></div>', unsafe_allow_html=True)
    ratings = api('GET', f'/accounts/{account_id}/ratings') or {'ratings': []}
    if len(ratings['ratings']) < 5:
        st.info(f'Rate {5 - len(ratings["ratings"])} more movie(s) to unlock your recommendations.')
        discover(account_id, onboarding=True)
        return
    result = api('POST', '/recommend', json={'account_id': account_id, 'top_k': 12})
    st.markdown('<h2 class="section">Picked for you</h2>', unsafe_allow_html=True)
    if result and result.get('recommendations'):
        cols = st.columns(3)
        for i, item in enumerate(result['recommendations']):
            with cols[i % 3]: movie_card(item, account_id)


def discover(account_id, onboarding=False):
    st.markdown('<div class="eyebrow">Discover</div><h1>Browse the catalog</h1>', unsafe_allow_html=True)
    local_tab, imdb_tab = st.tabs(['CineMatch catalog', 'IMDb search'])
    with imdb_tab:
        imdb_query = st.text_input('Search IMDb / OMDb', placeholder='Search any movie title', key='imdb-query')
        if imdb_query:
            external = api('GET', '/external/movies/search', params={'q': imdb_query})
            if external:
                cols = st.columns(4)
                for i, item in enumerate(external.get('movies', [])):
                    with cols[i % 4]:
                        poster = load_poster(item.get('poster', ''))
                        if poster: st.image(poster, width=140)
                        else: st.markdown('<div style="height:180px;width:120px;background:#252a37;border-radius:8px;display:flex;align-items:center;justify-content:center;color:#a6acba;font-size:.75rem;text-align:center;">No poster</div>', unsafe_allow_html=True)
                        st.markdown(f'**{item["title"]}**  \n{item.get("year", "")}')
                        if st.button('＋ Add to catalog', key=f'import-{item["imdb_id"]}'):
                            details = api('GET', f'/external/movies/{item["imdb_id"]}')
                            if details:
                                api('POST', '/catalog/import', json=details)
                                st.toast('Movie added to CineMatch')
    with local_tab:
        q = st.text_input('Search by title', placeholder='Search your imported and MovieLens movies')
        result = api('GET', '/movies/search', params={'q': q, 'limit': 50})
        if not result: return
        genres = ['All genres'] + result.get('genres', [])
        genre = st.selectbox('Filter by genre', genres)
        if genre != 'All genres': result = api('GET', '/movies/search', params={'q': q, 'genre': genre, 'limit': 50}) or result
        st.caption(f'{len(result.get("movies", []))} movies')
        cols = st.columns(3)
        for i, item in enumerate(result.get('movies', [])):
            with cols[i % 3]:
                movie_card(item, account_id)
                with st.expander('Rate / similar'):
                    rating = st.slider('Your rating', .5, 5., 4., .5, key=f'rate-{item["movieId"]}')
                    if st.button('Save rating', key=f'save-{item["movieId"]}'):
                        api('POST', '/ratings', json={'account_id': account_id, 'movie_id': item['movieId'], 'rating': rating})
                        st.toast('Taste updated'); st.rerun()
                    if st.button('Show similar', key=f'similar-{item["movieId"]}'):
                        similar = api('GET', f'/movies/{item["movieId"]}/similar')
                        if similar:
                            st.write('Because you may like:')
                            for match in similar['movies'][:4]: st.write(f'• {match["title"]}')


def watchlist(account_id):
    st.markdown('<div class="eyebrow">Saved for later</div><h1>Watchlist</h1>', unsafe_allow_html=True)
    result = api('GET', f'/accounts/{account_id}/watchlist')
    movies = result.get('movies', []) if result else []
    if not movies: st.info('Your watchlist is empty. Save films from Discover.'); return
    cols = st.columns(3)
    for i, item in enumerate(movies):
        with cols[i % 3]: movie_card(item, account_id, 'remove')


def profile(account_id):
    st.markdown('<div class="eyebrow">Your taste</div><h1>Profile</h1>', unsafe_allow_html=True)
    result = api('GET', f'/accounts/{account_id}/ratings') or {'ratings': []}
    ratings = result['ratings']
    c1, c2 = st.columns(2); c1.metric('Films rated', len(ratings)); c2.metric('Average rating', f'{sum(x["rating"] for x in ratings)/len(ratings):.1f}' if ratings else '—')
    if ratings:
        st.subheader('Your ratings')
        for item in ratings: st.write(f'★ {item["rating"]:.1f}  ·  {item["title"]}')


def main():
    if 'account' not in st.session_state:
        auth_screen(); return
    acc = st.session_state.account; account_id = acc['id']
    with st.sidebar:
        st.markdown('<div class="brand">CineMatch</div>', unsafe_allow_html=True)
        st.caption(f'Welcome, {acc["display_name"]}')
        page = st.radio('Navigate', ['Home', 'Discover', 'Watchlist', 'Profile'], label_visibility='collapsed')
        st.divider()
        if st.button('Sign out'):
            del st.session_state.account; st.rerun()
    if page == 'Home': home(account_id)
    elif page == 'Discover': discover(account_id)
    elif page == 'Watchlist': watchlist(account_id)
    else: profile(account_id)


if __name__ == '__main__': main()
