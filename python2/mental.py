import streamlit as st
import joblib

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------

st.set_page_config(
    page_title="MindShield",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------------------------------------------------
# HIDE STREAMLIT DEFAULT UI
# ---------------------------------------------------

st.markdown("""
<style>

#MainMenu {visibility:hidden;}
footer {visibility:hidden;}
header {visibility:hidden;}
[data-testid="collapsedControl"]{display:none;}

.block-container{
padding-top:0rem;
padding-bottom:2rem;
padding-left:0rem;
padding-right:0rem;
max-width:100%;
}

.stApp{
background:linear-gradient(180deg,#EEF9FF,#DDF2FF,#F7FCFF);
}

/* ---------- Titles ---------- */

.title{
font-size:68px;
font-weight:800;
color:#35627B;
text-align:center;
margin-top:60px;
}

.subtitle{

font-size:22px;

color:#64839B;

text-align:center;

margin-bottom:40px;

}

/* ---------- Welcome Card ---------- */

.hero{

width:78%;

margin:auto;

background:white;

padding:45px;

border-radius:35px;

box-shadow:0px 15px 35px rgba(0,0,0,.08);

text-align:center;

}

/* ---------- Robot ---------- */

.robot{

font-size:120px;

animation:float 3s ease-in-out infinite;

}

@keyframes float{

0%{transform:translateY(0px);}

50%{transform:translateY(-12px);}

100%{transform:translateY(0px);}

}

/* ---------- Buttons ---------- */

.stButton>button{

background:#A9D8FF;

color:#274B63;

font-weight:bold;

font-size:20px;

border:none;

border-radius:18px;

padding:15px;

transition:.3s;

}

.stButton>button:hover{

background:#7CC8FF;

transform:scale(1.02);

}

/* ---------- Cards ---------- */

.card{

background:white;

padding:30px;

border-radius:25px;

box-shadow:0px 10px 25px rgba(0,0,0,.08);

margin-top:20px;

}

</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SESSION STATE
# ---------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "confidence" not in st.session_state:
    st.session_state.confidence = 0

if "probabilities" not in st.session_state:
    st.session_state.probabilities = None
if "selected_option" not in st.session_state:
    st.session_state.selected_option = ""

# ---------------------------------------------------
# LOAD MODEL
# ---------------------------------------------------

@st.cache_resource
def load_model():
    return joblib.load("mental_health_model.pkl")

try:
    model = load_model()

except Exception as e:

    st.error(f"Unable to load model.\n\n{e}")

    st.stop()

# ---------------------------------------------------
# HOME PAGE
# ---------------------------------------------------

if st.session_state.page == "home":

    st.markdown(
        "<div class='title'>🧠 MindShield</div>",
        unsafe_allow_html=True
    )

    st.markdown(
        "<div class='subtitle'>Your AI Mental Wellness Companion</div>",
        unsafe_allow_html=True
    )

    st.markdown("""

<div class="hero">

<div class="robot">

🤖

</div>

<h2 style="color:#35627B;">

Welcome to MindShield

</h2>

<p style="font-size:20px;color:#64839B;line-height:1.8;">

A safe space where you can write your thoughts.

MindShield uses Artificial Intelligence and NLP
to understand your text and provide wellness insights,
personalized guidance and self-care recommendations.

</p>

</div>

""", unsafe_allow_html=True)

    st.write("")
    st.write("")

    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:

        if st.button("💙 Let's Get Started", use_container_width=True):

            st.session_state.page = "analysis"

            st.rerun()
# ---------------------------------------------------
# ANALYSIS PAGE
# ---------------------------------------------------

elif st.session_state.page == "analysis":

    # ---------- Back Button ----------

    col1, col2 = st.columns([1, 8])

    with col1:
        if st.button("⬅ Back"):
            st.session_state.page = "home"
            st.rerun()

    st.markdown("""

    <div class="card">

    <div style="text-align:center;font-size:90px;">
    🤖
    </div>

    <h2 style="text-align:center;color:#35627B;">
    Hello, I'm MindShield
    </h2>

    <p style="
    text-align:center;
    color:#64839B;
    font-size:18px;
    ">

    Write whatever is on your mind today.

    There are no right or wrong answers.

    I'm here to understand your thoughts.

    </p>

    </div>

    """, unsafe_allow_html=True)

    text = st.text_area(
        "",
        height=220,
        placeholder="Example: I feel lonely lately and I don't know why..."
    )

    if st.button("🩵 Analyze My Thoughts", use_container_width=True):

        if text.strip() == "":

            st.warning("Please write something first.")

        else:

            with st.spinner("MindShield is analysing your thoughts..."):

                prediction = model.predict([text])[0]

                probabilities = model.predict_proba([text])[0]

                confidence = round(probabilities.max() * 100, 2)

                st.session_state.prediction = prediction
                st.session_state.probabilities = probabilities
                st.session_state.confidence = confidence

            st.rerun()

    # -------------------------------------------------

    if st.session_state.prediction is not None:

        prediction = st.session_state.prediction
        confidence = st.session_state.confidence
        probs = st.session_state.probabilities

        colour = {

            "Normal":"#D8F7D4",

            "Depression":"#FFD7D7",

            "Anxiety":"#FFF3C7",

            "Stress":"#FFE5CC",

            "Bipolar":"#E8D9FF",

            "Personality disorder":"#D8E8FF"

        }

        bg = colour.get(prediction, "#D6EDFF")

        st.write("")
        st.write("")

        st.markdown(f"""

        <div style="
        background:{bg};
        padding:30px;
        border-radius:25px;
        box-shadow:0px 10px 25px rgba(0,0,0,.08);
        ">

        <h2 style="color:#305C75;">
        🧠 Diagnosis Result
        </h2>

        <h1 style="color:#24485E;">
        {prediction}
        </h1>

        <h3 style="color:#55758C;">
        Confidence : {confidence}%
        </h3>

        </div>

        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("## 📊 Mood Analytics")

        labels = model.classes_

        for label, value in zip(labels, probs):

            st.write(f"**{label}**")

            st.progress(int(value * 100))

            st.caption(f"{value*100:.2f}%")

        st.write("")

        if prediction == "Normal":

            st.success("💙 Your responses suggest a generally healthy emotional state. Continue maintaining healthy habits and emotional balance.")

        elif prediction == "Depression":

            st.info("💙 Try to maintain a routine, stay connected with people you trust, and take small manageable steps each day. If these feelings persist or become overwhelming, consider reaching out to a qualified mental health professional.")

        elif prediction == "Anxiety":

            st.info("💙 Slow breathing exercises, reducing caffeine, and taking breaks may help manage anxious feelings.")

        elif prediction == "Stress":

            st.info("💙 Try breaking tasks into smaller goals, staying hydrated, and giving yourself regular breaks.")

        elif prediction == "Bipolar":

            st.info("💙 Maintaining a consistent routine and following any professional treatment plan you may already have can be helpful.")

        else:

            st.info("💙 Building healthy routines, journaling, and discussing concerns with a qualified professional can be beneficial.")

        st.write("")

        if st.button("Know More ➜", use_container_width=True):

            st.session_state.page = "know_more"

            st.rerun()
# ---------------------------------------------------
# KNOW MORE PAGE
# ---------------------------------------------------

elif st.session_state.page == "know_more":

    left, right = st.columns([1,8])

    with left:
        if st.button("⬅ Back"):
            st.session_state.page = "analysis"
            st.rerun()

    st.markdown("""

    <div class="card">

    <h1 style="text-align:center;color:#35627B;">
    🌿 Know More
    </h1>

    <p style="
    text-align:center;
    color:#64839B;
    font-size:18px;
    ">

    Your analysis is complete.

    Continue with one of the options below.

    </p>

    </div>

    """,unsafe_allow_html=True)

    st.write("")

    col1,col2=st.columns(2)

    with col1:

        st.markdown("""

<div style="
background:#DDF3FF;
padding:35px;
border-radius:25px;
text-align:center;
box-shadow:0px 8px 20px rgba(0,0,0,.08);
">

<h1>🌱</h1>

<h2 style="color:#35627B;">
Self Care Hub
</h2>

<p style="color:#64839B;">

Receive a personalized daily checklist
based on your prediction.

</p>

</div>

""",unsafe_allow_html=True)

        if st.button(
            "Open Self Care Hub",
            use_container_width=True
        ):

            st.session_state.page="selfcare"

            st.rerun()

    with col2:

        st.markdown("""

<div style="
background:#EFE7FF;
padding:35px;
border-radius:25px;
text-align:center;
box-shadow:0px 8px 20px rgba(0,0,0,.08);
">

<h1>🛡️</h1>

<h2 style="color:#35627B;">
Responsible Use
</h2>

<p style="color:#64839B;">

Understand the purpose,
limitations,
and mental health resources.

</p>

</div>

""",unsafe_allow_html=True)

        if st.button(
            "Open Responsible Use",
            use_container_width=True
        ):

            st.session_state.page="responsible"

            st.rerun()
# ---------------------------------------------------
# SELF CARE HUB
# ---------------------------------------------------

elif st.session_state.page == "selfcare":

    left, right = st.columns([1, 8])

    with left:
        if st.button("⬅ Back"):
            st.session_state.page = "know_more"
            st.rerun()

    prediction = st.session_state.prediction

    st.markdown(f"""
    <div class="card">

    <h1 style="text-align:center;color:#35627B;">
    🌿 Self Care Hub
    </h1>

    <h3 style="text-align:center;color:#64839B;">
    Daily Checklist for <b>{prediction}</b>
    </h3>

    </div>
    """, unsafe_allow_html=True)

    routines = {

        "Normal":[
            "💧 Drink at least 2 litres of water",
            "🚶 Walk or exercise for 30 minutes",
            "🥗 Eat healthy meals",
            "📚 Learn something new",
            "👨‍👩‍👧 Spend time with family or friends",
            "📖 Read for 20 minutes",
            "😴 Sleep for 7–8 hours",
            "🙏 Practice gratitude"
        ],

        "Depression":[
            "🛏 Get out of bed and freshen up",
            "🥣 Eat one nutritious meal",
            "💧 Drink enough water",
            "☀ Spend 15 minutes in sunlight",
            "🚶 Take a short walk",
            "📞 Talk to someone you trust",
            "🎵 Listen to calming music",
            "📔 Write your thoughts in a journal",
            "😴 Sleep on time"
        ],

        "Anxiety":[
            "🌬 Practice deep breathing",
            "🧘 Try a grounding exercise",
            "☕ Reduce caffeine today",
            "🚶 Walk for 15 minutes",
            "💧 Stay hydrated",
            "🎵 Listen to relaxing music",
            "📝 Write down your worries",
            "📴 Avoid excessive screen time"
        ],

        "Stress":[
            "📋 Prioritize today's tasks",
            "💧 Drink water regularly",
            "🚶 Stretch every hour",
            "🌿 Take short breaks",
            "🎵 Listen to calming music",
            "📖 Read something enjoyable",
            "😴 Sleep before midnight",
            "😊 Do something you enjoy"
        ],

        "Bipolar":[
            "⏰ Follow a consistent routine",
            "😴 Maintain regular sleep",
            "🥗 Eat meals on time",
            "📔 Record today's mood",
            "🚫 Avoid alcohol or drugs",
            "👨‍⚕ Follow prescribed treatment if applicable",
            "👨‍👩‍👧 Connect with a trusted person",
            "🧘 Practice relaxation"
        ],

        "Personality disorder":[
            "🧘 Practice mindfulness",
            "📔 Journal today's emotions",
            "🚶 Go outside for fresh air",
            "🎨 Spend time on a hobby",
            "👥 Connect with someone supportive",
            "💧 Drink enough water",
            "📵 Reduce social media usage",
            "😴 Sleep well tonight"
        ]

    }

    tasks = routines.get(prediction, [])

    completed = 0

    st.write("")

    for i, task in enumerate(tasks):

        key = f"{prediction}_{i}"

        if st.checkbox(task, key=key):

            completed += 1

    st.write("")

    total = len(tasks)

    percent = int((completed/total)*100) if total else 0

    st.markdown("## 📊 Today's Progress")

    st.progress(percent)

    st.success(f"{completed} of {total} tasks completed ({percent}%)")

    st.write("")

    if percent == 100:

        st.balloons()

        st.success("Fantastic! You completed today's routine. 💙")

    elif percent >= 75:

        st.info("Excellent progress! Keep it up. 🌸")

    elif percent >= 40:

        st.info("You're making progress one step at a time. 🌿")

    else:

        st.info("Every small step counts. Start with just one task today. 💙")

    st.write("")

    st.markdown("""
    <div class="card">

    <h3 style="color:#35627B;">
    🌼 Daily Encouragement
    </h3>

    <p style="font-size:18px;color:#64839B;">

    Progress doesn't have to be perfect.
    Small consistent actions often make a meaningful difference over time.

    </p>

    </div>

    """, unsafe_allow_html=True)

    st.write("")

    c1, c2 = st.columns(2)

    with c1:

        if st.button("⬅ Back to Know More", use_container_width=True):

            st.session_state.page = "know_more"

            st.rerun()

    with c2:

        if st.button("🏠 Return Home", use_container_width=True):

            st.session_state.page = "home"

            st.session_state.prediction = None
            st.session_state.confidence = 0
            st.session_state.probabilities = None

            st.rerun()
# ---------------------------------------------------
# RESPONSIBLE USE PAGE
# ---------------------------------------------------

elif st.session_state.page == "responsible":

    left, right = st.columns([1,8])

    with left:
        if st.button("⬅ Back"):
            st.session_state.page = "know_more"
            st.rerun()

    st.markdown("""
    <div class="card">

    <h1 style="text-align:center;color:#35627B;">
    🛡 Responsible Use
    </h1>

    <p style="
    text-align:center;
    font-size:18px;
    color:#64839B;
    ">

    Please read the following information before interpreting your result.

    </p>

    </div>
    """,unsafe_allow_html=True)

    st.warning("""

### ⚠ Educational Purpose Only

MindShield is an AI-powered educational project.

It is designed to classify written text into mental-health related categories using Machine Learning.

It **does not replace** professional medical advice, diagnosis or treatment.

""")

    st.info("""

### 📌 Keep in Mind

• AI models can make mistakes.

• Your emotional state changes over time.

• One prediction should never define you.

• Use this application only as an awareness tool.

""")

    st.markdown("## 💙 If You Are Feeling Overwhelmed")

    st.success("""

You may consider:

• Talking to a trusted friend

• Speaking with a family member

• Taking a break from stressful situations

• Maintaining healthy sleep

• Eating regular meals

• Contacting a licensed mental-health professional if your concerns continue

""")

    st.markdown("## ☎ Mental Health Helplines (India)")

    st.markdown("""

### Tele-MANAS

📞 **14416**

📞 **1-800-891-4416**

Free Government Mental Health Helpline

""")

    st.markdown("""

### AASRA

📞 **+91 9820466726**

24×7 Emotional Support

""")

    st.error("""

### 🚨 Emergency

If you think you may act on thoughts of harming yourself,
or you believe you are in immediate danger,

please contact your local emergency services
or someone you trust immediately.

""")

    st.markdown("""

<div class="card">

<h2 style="text-align:center;color:#35627B;">
❤️ Thank You
</h2>

<p style="
text-align:center;
font-size:18px;
color:#64839B;
">

Thank you for using MindShield.

We hope this project encouraged you
to become more aware of your emotional wellbeing.

Remember:

Small positive habits every day
can make a meaningful difference over time.

Take care of yourself.

🌸

</p>

</div>

""",unsafe_allow_html=True)

    st.write("")

    c1,c2=st.columns(2)

    with c1:

        if st.button("⬅ Back to Know More",use_container_width=True):

            st.session_state.page="know_more"

            st.rerun()

    with c2:

        if st.button("🏠 Return Home",use_container_width=True):

            st.session_state.page="home"

            st.session_state.prediction=None
            st.session_state.confidence=0
            st.session_state.probabilities=None

            st.rerun()
