"""
Script to populate a ChromaDB vector database with book summaries and their embeddings.

This script creates a new ChromaDB collection and adds book summaries as documents,
generating embeddings using OpenAI's embedding model. The embeddings enable 
semantic similarity search across the summaries.
"""

import os
import sys
import uuid
import json
import dotenv
import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

# Load environment variables from .env file
dotenv.load_dotenv()

# Constants
COLLECTION_NAME = "bb_summaries"  # Name of the ChromaDB collection
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # OpenAI API key from environment
VECTOR_DB_PATH = "../vector_db"
# Book summaries to be added to the vector database
BOOK_SUMMARIES = [
    # Steve Jobs by Walter Isaacson
    "Walter Isaacson's 2011 biography, \"Steve Jobs,\" offers an in-depth exploration of the Apple co-founder's life and career. Based on over 40 interviews with Jobs and more than 100 conversations with family, friends, colleagues, and competitors, the book provides a comprehensive look at his personal and professional journey. It covers his early life, including his adoption and upbringing in Silicon Valley, his co-founding of Apple Inc., his departure and subsequent ventures like NeXT and Pixar, and his return to Apple, where he spearheaded the development of groundbreaking products such as the iMac, iPod, iPhone, and iPad. The biography delves into Jobs's complex personality, highlighting his visionary leadership, relentless pursuit of perfection, and often abrasive management style. Isaacson presents a balanced portrayal, acknowledging both Jobs's extraordinary achievements and his personal flaws, offering readers a nuanced understanding of a man who profoundly influenced the technology industry and modern culture.",
    # Meditations by Marcus Aurelius
    '"Meditations" is a collection of personal reflections by Roman Emperor Marcus Aurelius, written during his military campaigns between 170 and 180 AD. The work comprises 12 books, each containing insights into Stoic philosophy and guidance on virtuous living. Key themes include the transient nature of life, the importance of rationality, self-discipline, and the acceptance of events beyond one\'s control. Aurelius emphasizes focusing on one\'s own behavior, maintaining inner peace, and fulfilling duties to society. Originally intended for his personal use, "Meditations" offers timeless wisdom on resilience, ethical conduct, and personal growth.',
    # Život preduzetnika by Saša Popović
    '"Život preduzetnika" is an autobiographical work by Saša Popović, co-founder and CEO of Vega IT, a prominent IT company from Novi Sad. In this book, Popović shares his journey from modest beginnings, including working in construction and carrying sacks to support himself, to establishing a multimillion-dollar company. He delves into topics such as company building, business development, leadership, happiness, goodness, and personal fulfillment. The narrative offers insights into the challenges and lessons learned in entrepreneurship, emphasizing perseverance and the pursuit of one\'s goals. Notably, all proceeds from the book are donated to the Vega IT Foundation, which provides scholarships to talented young individuals in Serbia.',
    # Princip 80/20 by Richard Koch
    '"Princip 80/20" (originally "The 80/20 Principle") by Richard Koch explores the Pareto Principle, which posits that 80% of results stem from 20% of causes. Koch illustrates how this principle manifests across various domains: in business, where a minority of products or customers often generate the majority of profits; and in personal life, where a small fraction of activities contribute most to one\'s happiness and success. He advocates for identifying and focusing on these vital few inputs to enhance efficiency and effectiveness. The book provides practical strategies for applying the 80/20 Principle to time management, decision-making, and goal setting, aiming to help readers achieve more with less effort by concentrating on high-impact areas.',
    # From Zero to One by Peter Thiel
    '"From Zero to One" by Peter Thiel delves into the philosophy of innovation and entrepreneurship. Thiel argues that true progress comes not from copying others, but from creating something entirely new—moving from zero to one. The book focuses on the importance of building monopolies, not competing in crowded markets. Thiel emphasizes the significance of developing unique, game-changing technologies that have the potential to transform industries. He shares insights on creating successful startups, hiring the right people, and thinking differently about the future, with a strong focus on technological advancement and its impact on society.',
    #Elon Musk by Walter Isaacson
    '"Elon Musk" by Walter Isaacson is a biography that delves into the life of one of the most influential and controversial entrepreneurs of our time. The book chronicles Musk\'s journey from his early years in South Africa to his rise as the CEO of companies like Tesla, SpaceX, Neuralink, and The Boring Company. Isaacson explores Musk\'s vision for the future, including his drive to revolutionize space exploration, electric vehicles, and human-computer interaction. The biography also examines Musk\'s personal life, his relationships, and his often abrasive leadership style. Despite facing numerous setbacks and criticism, Musk\'s relentless ambition and ability to take risks have made him a key figure in shaping the future of technology. The book paints a complex portrait of a man driven by a desire to solve global challenges, often at great personal cost, while also highlighting his remarkable achievements in the fields of space, energy, and transportation.',
    #Great by Choice by Jim Collins  
    '"Great by Choice" by Jim Collins explores why some companies thrive in unpredictable and turbulent environments while others fail. Through extensive research, Collins and his team identify key principles that set successful companies apart, even in the face of uncertainty. The book introduces the concept of "10Xers" — leaders who outperform their competitors by a factor of 10, driven by a blend of fanatic discipline, empirical creativity, and productive paranoia. Collins emphasizes that great companies don\'t rely on luck but on disciplined decision-making, calculated risk-taking, and the ability to innovate while staying focused on long-term goals. Using case studies of companies like Southwest Airlines, Intel, and Amgen, Collins reveals the traits and behaviors that distinguish the truly great from the merely good. The book challenges the notion that success is determined by being bold and visionary, arguing instead that it is rooted in the ability to stay disciplined and resilient, regardless of external conditions.',
    #Emocionalna inteligencija by Daniel Goleman
    '"Emocionalna inteligencija" by Daniel Goleman explores the concept of emotional intelligence (EI), arguing that it plays a crucial role in personal success, relationships, and overall well-being, often surpassing the importance of traditional cognitive intelligence (IQ). Goleman identifies five key components of emotional intelligence: self-awareness, self-regulation, motivation, empathy, and social skills. He emphasizes that EI is not fixed but can be developed over time. The book explores how emotional intelligence influences various aspects of life, including leadership, decision-making, and interpersonal relationships. Goleman also highlights the significance of EI in education, parenting, and the workplace, advocating for a greater focus on emotional development alongside cognitive learning. He presents research findings that show how EI can be a better predictor of success in life than IQ, offering insights into how individuals can improve their emotional awareness and abilities to manage emotions effectively.',
    #Medvedi na putu by Bojan Lekić
    '"Medvedi na putu" by Bojan Lekić is a contemporary Serbian novel that delves into the complexities of human nature, relationships, and the search for meaning in a rapidly changing world. The story follows a protagonist who embarks on a journey of self-discovery while encountering personal and societal challenges. Lekić uses vivid imagery and metaphors, such as the recurring motif of bears on the road, to symbolize obstacles and struggles that individuals face in their quest for purpose. The novel explores themes of love, loss, identity, and the tension between tradition and modernity. Through its engaging narrative, Medvedi na putu invites readers to reflect on their own life paths and the choices they make along the way, while also offering a poignant commentary on the human condition and the ever-present search for meaning.',
    #Ciljevi by Brian Tracy
    '"Ciljevi" (originally "Goals") by Brian Tracy is a self-help book focused on the power of goal setting and achieving success. Tracy emphasizes the importance of setting clear, specific, and measurable goals, and outlines a structured approach to help readers achieve them. The book provides practical techniques for setting personal and professional goals, as well as strategies for overcoming obstacles, maintaining focus, and building momentum. Tracy introduces the concept of "success psychology," encouraging individuals to visualize their goals, break them down into smaller tasks, and develop the discipline required to stay on track. With actionable advice and motivational insights, Ciljevi serves as a guide for anyone looking to take control of their life, improve their productivity, and turn their dreams into reality. The book emphasizes that by setting and relentlessly pursuing well-defined goals, anyone can unlock their full potential and achieve lasting success.',
    #Power of Speech by Brian Tracy
    '"The Power of Speech" by Brian Tracy explores the significant role that effective communication plays in personal and professional success. Tracy explains how mastering the art of public speaking, persuasion, and negotiation can open doors to new opportunities, build credibility, and enhance leadership skills. The book covers key principles of powerful communication, including clarity, confidence, and emotional appeal, offering readers practical tips and techniques to improve their speaking abilities. Tracy also delves into the psychology of influence, showing how words can inspire, motivate, and persuade others to take action. By practicing the strategies outlined in Moć govora, individuals can elevate their careers, gain the trust of others, and become influential leaders in their field. The book emphasizes that anyone can become a more effective communicator by understanding the power of language and mastering the skills of persuasion and eloquence.',
    #Good to Great by Jim Collins
    '"Good to Great" by Jim Collins explores why some companies make the leap from good to great and sustain that success over time. Through extensive research and case studies, Collins identifies key factors that differentiate companies that achieve greatness from those that do not. One of the core concepts introduced is the "Hedgehog Concept," where companies focus on what they can be the best at, what drives their economic engine, and what they are passionate about. Collins also discusses the importance of Level 5 Leadership—leaders who are humble yet driven, with a commitment to building a great company rather than focusing on personal success. The book highlights the significance of disciplined people, disciplined thought, and disciplined action in driving long-term success. Collins presents the findings with practical advice on how businesses can transition from mediocrity to excellence, offering valuable insights for leaders and organizations aiming for sustainable greatness.',
    #Grad by Aleksandar Misojčić
    '"Grad: psihobiografija cara Dušana" by Aleksandar Misojčić is a unique exploration of the life and psyche of Emperor Dušan the Mighty, one of the most prominent rulers of medieval Serbia. In this book, Misojčić approaches Dušan\'s life through a psychological lens, offering a deep analysis of his personality, motivations, and the internal struggles that shaped his reign. The book delves into Dušan\'s complex character, examining his rise to power, his ambitions, and his vision for the Serbian Empire, while also exploring the human aspects of his leadership, such as his relationships with those around him and the pressures he faced. Misojčić combines historical facts with psychological insights, offering readers a comprehensive view of Dušan not only as a ruler but also as a person. The book also reflects on the broader political and social context of 14th-century Serbia, presenting an intriguing portrait of a ruler whose legacy is still debated today.',
    #Psihologija novca by Morgan Housel
    '"Psihologija novca" (The Psychology of Money) by Morgan Hausel explores the complex relationship people have with money and how psychological factors influence financial decisions. The book argues that financial success is not necessarily about intelligence or knowledge of markets, but about how individuals think and behave with money. Hausel examines how emotions, biases, and personal experiences shape our approach to saving, investing, and spending. He discusses topics such as the role of luck in wealth accumulation, the importance of long-term thinking, and how people\'s perceptions of risk and reward can drastically affect their financial outcomes. Through a series of stories and insights, the book challenges conventional wisdom and emphasizes the need for a thoughtful, patient, and disciplined approach to managing money. Hausel also highlights the importance of understanding our own financial psychology to make better, more informed decisions that align with our goals and values.',
    #Mit o preduzetništvu by Michael E. Gerber
    '"Mit o preduzetništvu" (The E-Myth Revisited) by Michael E. Gerber challenges common misconceptions about entrepreneurship and provides a roadmap for building a successful business. Gerber argues that many small business owners fail because they are technicians (skilled at a specific trade) rather than entrepreneurs (visionaries and managers). The book emphasizes the importance of working on the business, not just in it. Gerber introduces the concept of the "E-Myth," or the entrepreneurial myth, which suggests that being good at a craft does not automatically make someone good at running a business. He advocates for creating systems and processes that allow businesses to run efficiently without being overly dependent on the owner. By focusing on developing a scalable, replicable business model, Gerber provides practical advice for small business owners to grow their ventures into sustainable, long-term successes. The book is an essential guide for anyone looking to transform their passion into a thriving business by adopting the right mindset and operational strategies.',
    #Emocije by Zoran Milivojević
    '"Emocije" (Emotions) by Zoran Milivojević is a psychological exploration of human emotions and their impact on our lives. The book delves into the nature of emotions, their origins, and how they influence our behavior, relationships, and overall well-being. Milivojević explains the biological and psychological mechanisms behind emotions, highlighting how they shape our decisions and perceptions. He provides practical insights into emotional intelligence, offering tools for recognizing, understanding, and managing emotions effectively. The book also explores the role of emotions in personal development and how mastering emotional regulation can lead to better mental health, improved relationships, and greater success in life. Through a combination of theory and practical advice, Milivojević empowers readers to take control of their emotional lives, promoting emotional awareness as a key to personal growth and fulfillment.',
    #Psihologike by Zoran Milivojević
    '"Psihologike" by Zoran Milivojević is a book that examines various psychological principles and their application in daily life. It covers a wide range of topics, from understanding human behavior and emotions to the dynamics of relationships and personal development. Milivojević provides an insightful look into the human psyche, explaining how psychological theories can be translated into practical advice for improving one\'s life. The book touches on concepts like motivation, stress management, decision-making, and the role of cognitive biases in shaping our perceptions and actions. It aims to provide readers with the tools to better understand themselves and others, enhance communication skills, and navigate the complexities of modern life with a psychological perspective. Through clear explanations and real-life examples, Milivojević offers valuable insights into the functioning of the mind and its influence on our behavior and interactions.'
]   


# IDs for the book summaries in the database
IDXS = ["3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19"]



def main():
    """
    Main function to populate the ChromaDB vector database with book summaries.

    Creates a new collection (deleting existing one if present) and adds book summaries
    with their embeddings and metadata.
    """
    # Initialize OpenAI embedding function with API key
    embedding_function = OpenAIEmbeddingFunction(api_key=OPENAI_API_KEY)

    # Initialize ChromaDB client with persistent storage
    chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)

    # Delete existing collection if it exists
    try:
        chroma_client.delete_collection(name=COLLECTION_NAME)
        print("Deleted existing collection")
    except ValueError:
        pass

    # Create new collection with cosine similarity metric
    data_collection = chroma_client.create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_function,
        metadata={
            "hnsw:space": "cosine"
        },  # Use cosine similarity for vector comparisons
    )

    # Add each book summary to the collection
    for idx, book_summary in zip(IDXS, BOOK_SUMMARIES):
        print(f"Adding document {idx} with ID {idx}")
        data_collection.add(
            ids=[idx],
            documents=[book_summary],
            metadatas=[{"idx": idx}],  # Basic metadata storing the index
        )


if __name__ == "__main__":
    main()
