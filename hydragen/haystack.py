from pathlib import Path

from dataclasses import dataclass

import random

@dataclass
class Needle:
    sentence: str
    question: str
    answer: str
    position_in_doc: float

NEEDLES = []

animals = [
    "dog",
    "cat",
    "bird",
    "fish",
    "hamster",
    "rabbit",
    "snake",
    "turtle",
    "lizard",
    "frog",
    "toad",
]
names = [
    "Liam",
    "Noah",
    "Oliver",
    "James",
    "Elijah",
    "William",
    "Henry",
    "Lucas",
    "Benjamin",
    "Theodore",
    "Mateo",
    "Levi",
    "Sebastian",
    "Daniel",
    "Jack",
    "Michael",
    "Alexander",
    "Owen",
    "Asher",
    "Samuel",
    "Ethan",
    "Leo",
    "Jackson",
    "Mason",
    "Ezra",
    "John",
    "Hudson",
    "Luca",
    "Aiden",
    "Joseph",
    "David",
    "Jacob",
    "Logan",
    "Luke",
    "Julian",
    "Gabriel",
    "Grayson",
    "Wyatt",
    "Matthew",
    "Maverick",
    "Dylan",
    "Isaac",
    "Elias",
    "Anthony",
    "Thomas",
    "Jayden",
    "Carter",
    "Santiago",
    "Ezekiel",
    "Charles",
    "Josiah",
    "Caleb",
    "Cooper",
    "Lincoln",
    "Miles",
    "Christopher",
    "Nathan",
    "Isaiah",
    "Kai",
    "Joshua",
    "Andrew",
    "Angel",
    "Adrian",
    "Cameron",
    "Nolan",
    "Waylon",
    "Jaxon",
    "Roman",
    "Eli",
    "Wesley",
    "Aaron",
    "Ian",
    "Christian",
    "Ryan",
    "Leonardo",
    "Brooks",
    "Axel",
    "Walker",
    "Jonathan",
    "Easton",
    "Everett",
    "Weston",
    "Bennett",
    "Robert",
    "Jameson",
    "Landon",
    "Silas",
    "Jose",
    "Beau",
    "Micah",
    "Colton",
    "Jordan",
    "Jeremiah",
    "Parker",
    "Greyson",
    "Rowan",
    "Adam",
    "Nicholas",
    "Theo",
    "Xavier",
    "Hunter",
    "Dominic",
    "Jace",
    "Gael",
    "River",
    "Thiago",
    "Kayden",
    "Damian",
    "August",
    "Carson",
    "Austin",
    "Myles",
    "Amir",
    "Declan",
    "Emmett",
    "Ryder",
    "Luka",
    "Connor",
    "Jaxson",
    "Milo",
    "Enzo",
    "Giovanni",
    "Vincent",
    "Diego",
    "Luis",
    "Archer",
    "Harrison",
    "Kingston",
    "Atlas",
    "Jasper",
    "Sawyer",
    "Legend",
    "Lorenzo",
    "Evan",
    "Jonah",
    "Chase",
    "Bryson",
    "Adriel",
    "Nathaniel",
    "Arthur",
    "Juan",
    "George",
    "Cole",
    "Zion",
    "Jason",
    "Ashton",
    "Carlos",
    "Calvin",
    "Brayden",
    "Elliot",
    "Rhett",
    "Emiliano",
    "Ace",
    "Jayce",
    "Graham",
    "Max",
    "Braxton",
    "Leon",
    "Ivan",
    "Hayden",
    "Jude",
    "Malachi",
    "Dean",
    "Tyler",
    "Jesus",
    "Zachary",
    "Kaiden",
    "Elliott",
    "Arlo",
    "Emmanuel",
    "Ayden",
    "Bentley",
    "Maxwell",
    "Amari",
    "Ryker",
    "Finn",
    "Antonio",
    "Charlie",
    "Maddox",
    "Justin",
    "Judah",
    "Kevin",
    "Dawson",
    "Matteo",
    "Miguel",
    "Zayden",
    "Camden",
    "Messiah",
    "Alan",
    "Alex",
    "Nicolas",
    "Felix",
    "Alejandro",
    "Jesse",
    "Beckett",
    "Matias",
    "Tucker",
    "Emilio",
    "Xander",
    "Knox",
    "Oscar",
    "Beckham",
    "Timothy",
    "Abraham",
    "Andres",
    "Gavin",
    "Brody",
    "Barrett",
    "Hayes",
    "Jett",
    "Brandon",
    "Joel",
    "Victor",
    "Peter",
    "Abel",
    "Edward",
    "Karter",
    "Patrick",
    "Richard",
    "Grant",
    "Avery",
    "King",
    "Caden",
    "Adonis",
    "Riley",
    "Tristan",
    "Kyrie",
    "Blake",
    "Eric",
    "Griffin",
    "Malakai",
    "Rafael",
    "Israel",
    "Tate",
    "Lukas",
    "Nico",
    "Marcus",
    "Stetson",
    "Javier",
    "Colt",
    "Omar",
    "Simon",
    "Kash",
    "Remington",
    "Jeremy",
    "Louis",
    "Mark",
    "Lennox",
    "Callum",
    "Kairo",
    "Nash",
    "Kyler",
    "Dallas",
    "Crew",
    "Preston",
    "Paxton",
    "Steven",
    "Zane",
    "Ronald",
    "Bailey",
    "Marley",
]

colors = [
    "black",
    "white",
    "brown",
    "yellow",
    "orange",
    "red",
    "green",
    "blue",
]

random.seed(9)
seen_combos = set()
for name in names:
    color = random.choice(colors)
    sentence = f"The dog named {name} has fur that is {color}."
    question = f"What color is the fur of the dog named {name}?"
    answer = color.title()
    NEEDLES.append(Needle(sentence, question, answer, None))
random.shuffle(NEEDLES)

def make_needle_haystack(target_context_length: int, num_needles: int):
    """
    Construct a 'needle in the haystack' document by interleaving questions 
    in the text 'war and peace'.

    Args:
        target_context_length: approximate length of returned document in chars.
        num_needles: how many questions to put in the dataset.
    """
    assert num_needles > 2

    content_file = Path(__file__).parent.parent / "data" / "war_and_peace.txt"
    content = content_file.read_text()

    random.seed(9)
    
    needles = NEEDLES[:num_needles]

    results = [needles[0].sentence]

    # chars between needles
    approx_block_size = target_context_length // (num_needles - 1)

    content_idx_lower = 0
    content_idx_upper = approx_block_size

    positions = [0]

    for needle_index, needle in enumerate(needles):
        if needle_index == 0:
            continue

        # Move index until we get to end of a sentence
        while content[content_idx_upper] != ".":
            content_idx_upper += 1

        # content[content_idx_upper] is a .
        results.append(content[content_idx_lower : content_idx_upper + 1])

        positions.append(sum([len(x) for x in results]))
        results.append(needle.sentence)

        content_idx_lower = content_idx_upper + 1  # char after the period
        content_idx_upper += approx_block_size + 1

    content = " ".join(results).replace("\n", " ").replace("  ", " ").strip()

    for needle, position in zip(needles, positions):
        needle.position_in_doc = position / len(content)

    return content, needles

