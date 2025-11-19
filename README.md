# ExoplanetBot — Telegram-бот с локальным ИИ (Ollama)

Отвечает на любые вопросы об экзопланетах, открытых после 2021 года

---

* Понимает вопросы на естественном языке
→ «Расскажи про TOI-756 c»
→ «Какие суперземли открыли в 2024 году?»
→ «Сколько горячих юпитеров ближе 100 световых лет?»

* Ищет релевантные данные в актуальной базе NASA Exoplanet Archive (только планеты после 2021 года)

* Использует современный RAG (Retrieval-Augmented Generation):
– векторный поиск
– генерация ответа через локальный ИИ (Ollama)

* Работает везде, где есть Python 3.10+

---

## Структура проекта

```

ExoplanetBot-AI-local/
│
├── .env                  
├── data/
│   └── exoplanets_with_descriptions.csv   # база + эмбеддинги
├── main.py
├── requirements.txt
└── README.md

```

---

## Установка и запуск

1. Убедитесь, что установлен **Python 3.10+**
2. Убедитесь, что установлен **Ollama** с моделями **gpt-oss:20b** и **embeddinggemma** 
3. Создайте виртуальное окружение и активируйте его:

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
````

4. Установите зависимости:

```bash
pip install -r requirements.txt
```

5. В файле `.env` замените токен:

```python
TELEGRAM_TOKEN=YOUR_BOT_TOKEN
```

6. Запустите бота:

```bash
python main.py
```

---

## Технологии

* Python 3.10+
* openai
* aiogram
* pandas
* tiktoken
* scipy
* nest_asyncio
* python-dotenv
