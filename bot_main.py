from aiogram import Bot, Dispatcher, executor, types
import random

API_TOKEN = '6166359013:AAFqOSZk5nnDqAnYSdyonTvlJn28y_duAJw'

bot = Bot(token=API_TOKEN)
bot = Dispatcher(bot)

answers = {
    1: 'Go again?',
    2: "Let's do it again!",
    3: 'One more time?',
    4: 'Try once more!',
    5: 'Keep doing...',
    6: "So easy..."
}


@bot.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Hey human, if you give me a picture, I'll tell you what's in it.")


@bot.message_handler(content_types=['photo', 'document'])
async def get_picture(message: types.Message):
    await(message.answer("I know what it is, it's PIXELS.\n" + answers[random.randint(1, 5)]))


@bot.message_handler(content_types=['text'])
async def valid_input(message: types.Message):
    await(message.answer("PICTURE!!!"))


if __name__ == '__main__':
    executor.start_polling(bot, skip_updates=True)
