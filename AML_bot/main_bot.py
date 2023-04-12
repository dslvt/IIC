from aiogram import Bot, Dispatcher, executor, types
from PIL import Image
import open_clip
import torch
import Constants


bot = Bot(token=Constants.API_TOKEN)
dp = Dispatcher(bot)

model, _, transform = open_clip.create_model_and_transforms(
  model_name="coca_ViT-L-14",
  pretrained="weights/mscoco_finetuned_CoCa-ViT-L-14-laion2B-s13B-b90k.bin"
)


async def photo(message):
    file_id = message.document.file_id
    file = await bot.get_file(file_id)
    file_path = file.file_path
    await bot.download_file(file_path, "user_image.jpg")

    img = Image.open("user_image.jpg").convert("RGB")
    img = transform(img).unsqueeze(0)
    return img


async def generation(img):
    with torch.no_grad(), torch.cuda.amp.autocast():
        generated = model.generate(img)

    return open_clip.decode(generated[0]).split("<end_of_text>")[0].replace("<start_of_text>", "")


@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Hey human, if you give me a picture, I'll tell you what's in it.")


@dp.message_handler(content_types=['document'])
async def get_picture(message: types.Message):
    img = await (photo(message))
    text = await (generation(img))
    await(message.answer(text))


@dp.message_handler(content_types=['photo'])
async def get_picture(message: types.Message):
    await(message.answer('Please, load it as a document'))

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
