"""esl_story_generator.py – ESL (English as a Second Language) story generator.

Generates short English learning stories following ESL teaching guidelines:

- 6 short sentences (maximum 10 words each)
- Natural and conversational language
- Everyday or professional context
- Suitable for adult learners
- No translations or phonetics

Levels supported: A2, B1, B2, C1 (CEFR framework)
"""

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CEFR level definitions
# ---------------------------------------------------------------------------

_VALID_LEVELS = frozenset({"A2", "B1", "B2", "C1"})

# ---------------------------------------------------------------------------
# Per-topic, per-level story templates
# Each list contains exactly 6 sentences of at most 10 words.
# ---------------------------------------------------------------------------

_STORY_TEMPLATES: dict[str, dict[str, list[str]]] = {
    "work": {
        "A2": [
            "She goes to work at nine every morning.",
            "Her office is in the center of town.",
            "She talks to her boss about her tasks.",
            "They work together on a big project.",
            "She finishes work at five in the afternoon.",
            "She feels happy when she does a good job.",
        ],
        "B1": [
            "Maria has worked at the company for two years.",
            "She usually handles customer emails in the morning.",
            "Her team has just finished an important project.",
            "She would like to get a promotion soon.",
            "Her manager gave her positive feedback last week.",
            "She is thinking about taking a leadership course.",
        ],
        "B2": [
            "After joining the firm, Sarah quickly proved her value.",
            "She managed the project independently, meeting every deadline.",
            "Her colleagues relied on her communication and problem-solving skills.",
            "She proposed a new strategy to improve team efficiency.",
            "The company adopted her idea, leading to better results.",
            "Her professional growth was recognized with a well-deserved promotion.",
        ],
        "C1": [
            "Having navigated several restructurings, James understood office politics well.",
            "He leveraged his expertise to drive meaningful organizational change.",
            "His ability to articulate complex ideas set him apart.",
            "Colleagues often sought his counsel before critical business decisions.",
            "He mentored junior staff, fostering a culture of excellence.",
            "His work earned him a reputation as an industry leader.",
        ],
    },
    "travel": {
        "A2": [
            "Tom wants to visit a new city this summer.",
            "He buys a ticket at the train station.",
            "He packs his bag with clothes and a map.",
            "He arrives at the hotel in the evening.",
            "The next day, he visits a famous museum.",
            "He takes photos to remember the trip.",
        ],
        "B1": [
            "Ana has always dreamed of travelling to Japan.",
            "She saved money for a whole year to go.",
            "When she arrived, everything felt different and exciting.",
            "She tried local food and learned a few phrases.",
            "The temples and gardens left a lasting impression on her.",
            "She came back with wonderful memories and new friends.",
        ],
        "B2": [
            "Travelling solo requires careful planning and a flexible mindset.",
            "She researched local customs before arriving in Southeast Asia.",
            "Unexpected delays forced her to adapt her itinerary quickly.",
            "She discovered hidden gems by talking to local residents.",
            "Each destination broadened her perspective in unexpected ways.",
            "She returned with a richer understanding of global cultures.",
        ],
        "C1": [
            "Immersive travel challenges assumptions about one's own cultural identity.",
            "He avoided tourist hubs to engage with authentic communities.",
            "Navigating linguistic barriers sharpened his improvisational communication skills.",
            "Each encounter with locals revealed the nuances of daily life.",
            "He documented his reflections to process each experience deeply.",
            "The journey transformed both his worldview and his priorities.",
        ],
    },
    "food": {
        "A2": [
            "Lisa cooks dinner for her family every night.",
            "Tonight she makes pasta with tomato sauce.",
            "She cuts vegetables and adds them to the pan.",
            "The food smells very good in the kitchen.",
            "Her family sits at the table together.",
            "They all say the meal is delicious.",
        ],
        "B1": [
            "Carlos loves trying food from different countries.",
            "He recently learned to cook a Thai curry.",
            "He had to find some unusual spices at the market.",
            "The recipe was not easy, but he followed each step.",
            "His friends were impressed when they tasted the result.",
            "Now he cooks a new dish from abroad every month.",
        ],
        "B2": [
            "The relationship between food and culture is deeply intertwined.",
            "She explored regional cuisines during her year abroad in Italy.",
            "Each dish told a story rooted in local history.",
            "She adapted traditional recipes to suit her dietary preferences.",
            "Her experimentation in the kitchen led to creative fusion dishes.",
            "Food became her primary lens for understanding new cultures.",
        ],
        "C1": [
            "Gastronomy has evolved into a sophisticated form of cultural expression.",
            "He meticulously sourced ingredients from artisan producers across the region.",
            "The interplay of textures and flavors defined his culinary signature.",
            "He challenged conventional techniques to craft unexpected sensory experiences.",
            "Critics praised his menu for its conceptual depth and execution.",
            "His philosophy centered on honoring ingredients rather than masking them.",
        ],
    },
    "technology": {
        "A2": [
            "Jake uses his phone to read the news.",
            "He also sends messages to his friends every day.",
            "He watches videos on his computer at home.",
            "His sister taught him how to use a new app.",
            "Now he can pay for things with his phone.",
            "He thinks technology makes everyday life much easier.",
        ],
        "B1": [
            "Technology has changed the way people communicate at work.",
            "Many companies now use video calls instead of meetings.",
            "Employees can share files and ideas quickly online.",
            "Some workers prefer working from home with digital tools.",
            "Learning new software has become part of most jobs.",
            "These changes have made teams more flexible and efficient.",
        ],
        "B2": [
            "Artificial intelligence is reshaping industries at an unprecedented pace.",
            "Companies invest heavily in automation to reduce operational costs.",
            "Workers must continuously update their skills to remain competitive.",
            "Digital literacy has become an essential workplace requirement.",
            "The ethical implications of AI demand urgent public debate.",
            "Balancing innovation with human oversight is the key challenge ahead.",
        ],
        "C1": [
            "The convergence of AI and biotechnology is redefining human capability.",
            "Algorithmic decision-making raises profound questions about accountability.",
            "He argued that progress must be accompanied by ethical frameworks.",
            "Data privacy concerns are increasingly shaping legislative priorities globally.",
            "The digital divide threatens to deepen socioeconomic inequalities worldwide.",
            "Addressing these challenges requires interdisciplinary collaboration across sectors.",
        ],
    },
    "health": {
        "A2": [
            "Maria goes for a walk every morning.",
            "She drinks a lot of water during the day.",
            "She eats fruits and vegetables with every meal.",
            "She goes to bed early to get enough sleep.",
            "She feels strong and full of energy.",
            "Her doctor says she is in very good health.",
        ],
        "B1": [
            "Regular exercise can have a positive effect on mood.",
            "Doctors recommend at least thirty minutes of activity daily.",
            "A balanced diet helps the body stay strong and healthy.",
            "Getting enough sleep is just as important as eating well.",
            "Stress can affect both mental and physical wellbeing.",
            "Small daily habits can lead to big health improvements.",
        ],
        "B2": [
            "Mental health awareness has grown significantly in recent years.",
            "Many workplaces now offer support programs for stressed employees.",
            "Research links chronic stress to a range of physical conditions.",
            "Preventive healthcare reduces long-term costs for individuals and systems.",
            "Lifestyle choices made in youth can impact health decades later.",
            "A holistic approach integrates mind, body, and environment effectively.",
        ],
        "C1": [
            "The medicalization of everyday life raises critical sociological questions.",
            "He examined how systemic inequalities perpetuate health disparities globally.",
            "Precision medicine promises treatments tailored to individual genetic profiles.",
            "Public health policy must navigate competing economic and ethical interests.",
            "Behavioral science offers nuanced insights into sustainable lifestyle change.",
            "Advancing health equity demands structural reform, not merely individual action.",
        ],
    },
    "education": {
        "A2": [
            "Anna goes to school five days a week.",
            "Her favourite subject is English.",
            "She reads a short story in class every day.",
            "Her teacher helps her when she makes a mistake.",
            "After school, she does her homework at home.",
            "She loves learning new words in English.",
        ],
        "B1": [
            "Learning a new language takes time and practice.",
            "David studies English for thirty minutes every evening.",
            "He watches English films to improve his listening skills.",
            "He also writes short texts to practise grammar.",
            "He has made a lot of progress since last year.",
            "His teacher says he is ready for the next level.",
        ],
        "B2": [
            "Effective language learning involves more than memorizing vocabulary.",
            "She immersed herself in English by reading authentic texts.",
            "She kept a journal to reflect on her daily progress.",
            "Regular conversation practice helped her gain confidence quickly.",
            "She used feedback from native speakers to refine her accuracy.",
            "Her structured approach allowed her to reach fluency faster.",
        ],
        "C1": [
            "Acquiring advanced proficiency demands deliberate and sustained cognitive effort.",
            "He critically engaged with complex texts to deepen his comprehension.",
            "Nuanced vocabulary acquisition came through extensive contextual reading.",
            "He refined his style by studying academic and literary prose.",
            "Peer discussion challenged him to articulate abstract ideas precisely.",
            "His near-native fluency reflected years of disciplined, purposeful study.",
        ],
    },
    "shopping": {
        "A2": [
            "Sara goes to the supermarket on Saturday morning.",
            "She makes a list before she leaves home.",
            "She buys bread, milk, fruit, and vegetables.",
            "She pays at the cash register with her card.",
            "The cashier gives her a receipt.",
            "She puts the bags in the car and drives home.",
        ],
        "B1": [
            "Online shopping has become very popular in recent years.",
            "Tom usually compares prices before he buys anything.",
            "He reads reviews to make sure the product is good.",
            "He adds items to his cart and pays online.",
            "The package arrives at his door two days later.",
            "He is happy because everything was exactly as described.",
        ],
        "B2": [
            "Consumer behaviour has shifted dramatically towards e-commerce platforms.",
            "She carefully evaluates product ratings and user reviews before buying.",
            "Subscription services offer convenience but can lead to overspending.",
            "She sets a monthly budget to avoid impulse purchases.",
            "Sustainable shopping choices are becoming increasingly important to her.",
            "She prefers brands that align with her personal values.",
        ],
        "C1": [
            "Consumerism increasingly shapes cultural identity in modern societies.",
            "He questioned whether his purchasing habits reflected genuine needs.",
            "The psychology of scarcity drives much of retail marketing strategy.",
            "He advocated for conscious consumption as social responsibility.",
            "Fair trade became a key purchasing criterion for him.",
            "His choices reflected a deliberate rejection of disposable consumer culture.",
        ],
    },
}

# ---------------------------------------------------------------------------
# Generic fallback templates (used when the topic has no specific entry)
# Each sentence contains a {topic} placeholder.
# ---------------------------------------------------------------------------

_GENERIC_TEMPLATES: dict[str, list[str]] = {
    "A2": [
        "Mark is learning about {topic} for the first time.",
        "He reads a short book about it at school.",
        "He asks his teacher many questions about {topic}.",
        "His teacher gives him simple and clear answers.",
        "He practises what he learns every day at home.",
        "Now he knows a lot more about {topic}.",
    ],
    "B1": [
        "Learning about {topic} has always interested Sarah.",
        "She started studying it seriously about six months ago.",
        "She found several useful books and online resources.",
        "She practises regularly and tracks her own progress.",
        "Her knowledge has improved a lot since she began.",
        "She hopes to use these skills in her career soon.",
    ],
    "B2": [
        "A deep understanding of {topic} requires consistent, focused effort.",
        "She explored multiple perspectives before forming her own views.",
        "Practical experience proved just as valuable as theoretical knowledge.",
        "She applied what she learned to real professional challenges.",
        "Her growing expertise opened new opportunities she had not expected.",
        "She continues to refine her understanding through ongoing reflection.",
    ],
    "C1": [
        "Mastery of {topic} demands both intellectual curiosity and discipline.",
        "He critically evaluated competing theories before drawing any conclusions.",
        "He synthesized insights from diverse fields for a nuanced perspective.",
        "His expertise allowed him to challenge conventional assumptions confidently.",
        "He communicated complex ideas with clarity and persuasive precision.",
        "His contributions to the field earned wide recognition among peers.",
    ],
}

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_MAX_WORDS_PER_SENTENCE = 10


def _word_count(sentence: str) -> int:
    """Return the number of whitespace-separated words in *sentence*."""
    return len(sentence.split())


def _validate_sentences(sentences: list[str]) -> None:
    """Raise *ValueError* if any sentence exceeds the word limit."""
    for idx, sentence in enumerate(sentences, start=1):
        count = _word_count(sentence)
        if count > _MAX_WORDS_PER_SENTENCE:
            raise ValueError(
                f"Sentence {idx} has {count} words (limit is {_MAX_WORDS_PER_SENTENCE}): "
                f"'{sentence}'"
            )


def _get_story_sentences(topic: str, level: str) -> list[str]:
    """Return 6 sentences appropriate for *topic* and *level*.

    Looks up the topic in :data:`_STORY_TEMPLATES` (case-insensitive, partial
    match).  Falls back to :data:`_GENERIC_TEMPLATES` when the topic is not
    found.
    """
    normalised_topic = topic.strip().lower()

    # Try to find a matching topic key (substring match in either direction)
    for key in _STORY_TEMPLATES:
        if key in normalised_topic or normalised_topic in key:
            level_stories = _STORY_TEMPLATES[key]
            if level in level_stories:
                return list(level_stories[level])

    # Fall back to generic template with {topic} substituted
    return [s.format(topic=topic.strip()) for s in _GENERIC_TEMPLATES[level]]


def _format_story(sentences: list[str]) -> str:
    """Format a list of sentences as a numbered ESL story string."""
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_esl_story(topic: str, level: str) -> str:
    """Generate a short ESL story for adult learners.

    Creates a 6-sentence English story following ESL teaching guidelines
    as described in the prompt template below::

        Actúa como un profesor experto en ESL (English as a Second Language).

        Crea una historia corta de aprendizaje de inglés con estas reglas:
        - Tema: [TEMA]
        - Nivel: [A2 / B1 / B2 / C1]
        - 6 oraciones cortas
        - Máximo 10 palabras por oración
        - Lenguaje natural y conversacional
        - Contexto cotidiano o profesional
        - Debe ser útil para estudiantes adultos

        NO agregues traducciones.
        NO agregues fonética.
        Solo devuelve las 6 oraciones en inglés.

    Args:
        topic: The story topic in any language (e.g. ``"work"``, ``"travel"``,
               ``"food"``, ``"technology"``).
        level: CEFR level – one of ``"A2"``, ``"B1"``, ``"B2"``, ``"C1"``.

    Returns:
        A newline-separated string of 6 numbered English sentences::

            1. She goes to work at nine every morning.
            2. Her office is in the center of town.
            3. She talks to her boss about her tasks.
            4. They work together on a big project.
            5. She finishes work at five in the afternoon.
            6. She feels happy when she does a good job.

    Raises:
        ValueError: If *level* is not one of the supported CEFR levels, or if
                    *topic* / *level* are empty strings.
    """
    if not topic or not topic.strip():
        raise ValueError("'topic' must be a non-empty string.")

    normalised_level = level.strip().upper() if level else ""

    if normalised_level not in _VALID_LEVELS:
        raise ValueError(
            f"Invalid level '{level}'. "
            f"Valid levels are: {', '.join(sorted(_VALID_LEVELS))}"
        )

    sentences = _get_story_sentences(topic, normalised_level)
    story = _format_story(sentences)

    logger.info(
        "ESL story generated – topic: '%s', level: %s, sentences: %d",
        topic.strip(),
        normalised_level,
        len(sentences),
    )

    return story
