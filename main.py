import asyncio
import nest_asyncio
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters import Command
import pandas as pd
import ast
from openai import OpenAI
from scipy import spatial
import tiktoken
from aiogram.utils.keyboard import ReplyKeyboardBuilder
import os
from dotenv import load_dotenv

load_dotenv()

nest_asyncio.apply()

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Выбираем модель GPT
GPT_MODEL = "gpt-oss:20b"
EMBEDDING_MODEL = "embeddinggemma"

# Инициализация бота и диспетчера
bot = Bot(token=TELEGRAM_TOKEN)
dp = Dispatcher()

# Инициализация OpenAI
openai = OpenAI(
    base_url='http://localhost:11434/api',
    api_key=OPENAI_API_KEY,
)

# Путь к CSV с данными экзопланет
embeddings_path = "data/exoplanets_with_descriptions.csv"
df = pd.read_csv(embeddings_path)

# Добавляем колонку 'text' если её нет (имя + описание)
if 'text' not in df.columns:
    df['text'] = df['pl_name'].astype(str) + ": " + df['description'].astype(str)

# Генерация эмбеддингов, если колонки 'embedding' нет
if 'embedding' not in df.columns:
    def get_embedding(text):
        response = openai.embeddings.create(
            input=text,
            model=EMBEDDING_MODEL
        )
        return response.data[0].embedding

    df['embedding'] = df['text'].apply(get_embedding)
    # Сохраняем с эмбеддингами для будущего использования
    df.to_csv(embeddings_path, index=False)
else:
    # Конвертируем эмбеддинги из строк в списки, если они сохранены как строки
    df['embedding'] = df['embedding'].apply(ast.literal_eval)

print("Эмбеддинги готовы для", len(df), "записей")

# Функция поиска
def strings_ranked_by_relatedness(
    query: str, 
    df: pd.DataFrame, 
    relatedness_fn=lambda x, y: 1 - spatial.distance.cosine(x, y),
    top_n: int = 5
) -> tuple[list[str], list[float]]:

    query_embedding_response = openai.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    query_embedding = query_embedding_response.data[0].embedding

    strings_and_relatednesses = [
        (row["text"], relatedness_fn(query_embedding, row["embedding"]))
        for i, row in df.iterrows()
    ]

    strings_and_relatednesses.sort(key=lambda x: x[1], reverse=True)
    strings, relatednesses = zip(*strings_and_relatednesses)
    return strings[:top_n], relatednesses[:top_n]

# Функция для подсчёта токенов
def num_tokens(text: str, model: str = GPT_MODEL) -> int:
    encoding = tiktoken.get_encoding("cl100k_base")
    return len(encoding.encode(text))

# Функция формирования запроса к GPT
def query_message(
    query: str,
    df: pd.DataFrame,
    model: str,
    token_budget: int
) -> str:
    strings = strings_ranked_by_relatedness(query, df)
    message = 'Use the below data on exoplanets discovered after 2021 to answer the subsequent question. If the answer cannot be found in the data, write "I could not find an answer."'
    question = f"\n\nQuestion: {query}"

    for string in strings:
        next_article = f'\n\nExoplanet data section:\n"""\n{string}\n"""'
        if (num_tokens(message + next_article + question, model=model) > token_budget):
            break
        else:
            message += next_article
    return message + question

# Функция для получения ответа от GPT
def ask(
    query: str,
    df: pd.DataFrame = df,
    model: str = GPT_MODEL,
    token_budget: int = 4096 - 500,
    print_message: bool = False,
) -> str:
    message = query_message(query, df, model=model, token_budget=token_budget)
    if print_message:
        print(message)
    messages = [
        {"role": "system", "content": "You answer questions about exoplanets discovered after 2021 in Russian."},
        {"role": "user", "content": message},
    ]
    response = openai.chat.completions.create(
        model=model,
        messages=messages,
    )
    response_message = response.choices[0].message.content
    return response_message

@dp.message(Command('start'))
async def send_welcome(message: types.Message):
    builder = ReplyKeyboardBuilder()
    builder.add(types.KeyboardButton(text="Помощь"))
    builder.adjust(1)

    await message.answer(
        "Добро пожаловать! Этот бот отвечает на вопросы об экзопланетах, открытых после 2021 года, используя базу данных NASA и ИИ. Задайте вопрос, например: 'Расскажи о TOI-756 c'.",
        reply_markup=builder.as_markup(resize_keyboard=True)
    )

# Обработчик команды /help
@dp.message(F.text=="Помощь")
@dp.message(Command('help'))
async def send_help(message: types.Message):
    num_records = len(df)
    help_text = f"Информация о базе знаний:\n" \
                f"- Тематика: Экзопланеты, открытые после 2021 года (данные из NASA Exoplanet Archive).\n" \
                f"- Число записей: {num_records}.\n" \
                f"- Пример запроса: 'Расскажи о экзопланете TOI-756 c' или 'Какие экзопланеты открыты в 2025 году?'."
    await message.reply(help_text)

# Обработчик текстовых сообщений (для вопросов)
@dp.message()
async def handle_message(message: types.Message):
    if message.text.startswith('/'):
        return
    query = message.text
    try:
        # Используем функцию ask для получения ответа
        answer = ask(query)
        await message.reply(answer)
    except Exception as e:
        await message.reply(f"Ошибка: {str(e)}. Попробуйте позже.")

async def main():
    await dp.start_polling(bot)

if __name__ == '__main__':
    print("Бот запущен")
    asyncio.run(main())