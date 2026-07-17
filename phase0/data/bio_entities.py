"""Real, well-known people used as decomposition source subjects.

FActScore's original release (Min et al. 2023) was not found as a ready-made
HF Hub dataset (see probe attempts, 2026-07-13). For Phase 0's mechanistic
pre-validation we substitute self-generated closed-book biographies about
real, Wikipedia-notable people spanning several domains -- adequate for
testing whether the NQ-Swap-trained probe transfers to decomposition-induced
spans. Phase 1 (the main study) uses the actual FActScore release instead.

2026-07-15: expanded from 40 to ~190 people for H3 (CAD mitigation
experiment, see README.md). H3 needs a
paired before/after comparison on the same entity set, and the original
40-person pool only yielded 14 baseline DI-CC cases (35 had a perturbable
year) -- too few to detect a reduction with any power. Expansion keeps the
same "real, Wikipedia-notable, specific dateable facts" criterion, adds
domain/region breadth, and deliberately excludes founding religious figures
(Buddha, Jesus, etc.) since their birth years are contested/unknowable,
which breaks the SPECIFIC_BIO_PROMPT "at least one specific year" mechanism.
"""

BIO_ENTITIES = [
    # --- original Phase 0 pool (40) ---
    "Albert Einstein", "Marie Curie", "Isaac Newton", "Charles Darwin",
    "Ada Lovelace", "Alan Turing", "Nikola Tesla", "Rosalind Franklin",
    "Abraham Lincoln", "Winston Churchill", "Nelson Mandela", "Mahatma Gandhi",
    "Cleopatra", "Julius Caesar", "Napoleon Bonaparte", "Genghis Khan",
    "William Shakespeare", "Leonardo da Vinci", "Wolfgang Amadeus Mozart",
    "Ludwig van Beethoven", "Pablo Picasso", "Frida Kahlo", "Jane Austen",
    "Ernest Hemingway", "Virginia Woolf", "Martin Luther King Jr.",
    "Rosa Parks", "Malala Yousafzai", "Steve Jobs", "Bill Gates",
    "Elon Musk", "Marie Antoinette", "Queen Elizabeth II", "Confucius",
    "Aristotle", "Plato", "Galileo Galilei", "Stephen Hawking",
    "Neil Armstrong", "Amelia Earhart",

    # --- science / math (30) ---
    "Johannes Kepler", "Michael Faraday", "James Clerk Maxwell",
    "Werner Heisenberg", "Niels Bohr", "Max Planck", "Erwin Schrodinger",
    "Richard Feynman", "Enrico Fermi", "Dmitri Mendeleev", "Gregor Mendel",
    "Louis Pasteur", "Alexander Fleming", "Carl Linnaeus", "Alfred Nobel",
    "Srinivasa Ramanujan", "Carl Friedrich Gauss", "Leonhard Euler",
    "Blaise Pascal", "Rene Descartes", "Archimedes", "Pythagoras", "Euclid",
    "Hypatia", "Katherine Johnson", "Jane Goodall", "Rachel Carson",
    "Barbara McClintock", "Chien-Shiung Wu", "Tu Youyou",

    # --- technology / invention (11) ---
    "Thomas Edison", "Alexander Graham Bell", "Henry Ford", "Orville Wright",
    "Wilbur Wright", "Guglielmo Marconi", "Grace Hopper", "John von Neumann",
    "Tim Berners-Lee", "Steve Wozniak", "Jeff Bezos",

    # --- exploration (9) ---
    "Christopher Columbus", "Ferdinand Magellan", "Marco Polo", "Zheng He",
    "Vasco da Gama", "Ernest Shackleton", "Edmund Hillary", "Yuri Gagarin",
    "Sally Ride",

    # --- politics / leaders (24) ---
    "George Washington", "Thomas Jefferson", "Franklin D. Roosevelt",
    "Theodore Roosevelt", "John F. Kennedy", "Barack Obama",
    "Margaret Thatcher", "Angela Merkel", "Otto von Bismarck",
    "Vladimir Lenin", "Joseph Stalin", "Mao Zedong", "Sun Yat-sen",
    "Deng Xiaoping", "Indira Gandhi", "Jawaharlal Nehru", "Golda Meir",
    "Simon Bolivar", "Mustafa Kemal Ataturk", "Charlemagne",
    "Queen Victoria", "Henry VIII", "Louis XIV", "Catherine the Great",

    # --- military (7) ---
    "Alexander the Great", "Hannibal", "Sun Tzu", "Erwin Rommel",
    "Dwight D. Eisenhower", "George S. Patton", "Joan of Arc",

    # --- literature (16) ---
    "Mark Twain", "Charles Dickens", "Leo Tolstoy", "Fyodor Dostoevsky",
    "Homer", "Dante Alighieri", "Victor Hugo", "Franz Kafka",
    "George Orwell", "J.R.R. Tolkien", "Agatha Christie", "Toni Morrison",
    "Gabriel Garcia Marquez", "Maya Angelou", "Emily Dickinson",
    "Edgar Allan Poe",

    # --- visual arts (7) ---
    "Michelangelo", "Rembrandt", "Vincent van Gogh", "Claude Monet",
    "Salvador Dali", "Andy Warhol", "Georgia O'Keeffe",

    # --- music (10) ---
    "Johann Sebastian Bach", "Franz Schubert", "Frederic Chopin",
    "Pyotr Ilyich Tchaikovsky", "Richard Wagner", "Elvis Presley",
    "John Lennon", "Bob Dylan", "Michael Jackson", "Freddie Mercury",

    # --- film (5) ---
    "Charlie Chaplin", "Alfred Hitchcock", "Walt Disney", "Marilyn Monroe",
    "Audrey Hepburn",

    # --- philosophy (4) ---
    "Socrates", "Immanuel Kant", "Friedrich Nietzsche", "Karl Marx",

    # --- sports (7) ---
    "Muhammad Ali", "Pele", "Michael Jordan", "Serena Williams",
    "Usain Bolt", "Babe Ruth", "Jesse Owens",

    # --- business (4) ---
    "Warren Buffett", "Coco Chanel", "Andrew Carnegie", "J.P. Morgan",

    # --- activism / civil rights (6) ---
    "Frederick Douglass", "Harriet Tubman", "Susan B. Anthony",
    "Emmeline Pankhurst", "Desmond Tutu", "Aung San Suu Kyi",

    # --- global / regional breadth (10) ---
    "Haile Selassie", "Kwame Nkrumah", "Jomo Kenyatta", "Benazir Bhutto",
    "Rabindranath Tagore", "Akbar the Great", "Saladin", "Ibn Khaldun",
    "Averroes", "Avicenna",

    # --- documented modern history (1) ---
    "Anne Frank",
]
