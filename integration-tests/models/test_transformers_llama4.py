# import base64
# from io import BytesIO
# from PIL import Image
#
# import pytest
#
#
# @pytest.fixture(scope="module")
# def flash_llama4_handle(launcher):
#     with launcher("ll-re/Llama-4-Scout-17B-16E-Instruct", num_shard=8) as handle:
#         yield handle
#
#
# @pytest.fixture(scope="module")
# async def flash_llama4(flash_llama4_handle):
#     await flash_llama4_handle.health(300)
#     return flash_llama4_handle.client
#
#
# async def test_flash_llama4(flash_llama4, response_snapshot):
#     response = await flash_llama4.generate(
#         "Hello I am doing a project on the 1918 flu pandemic and I am trying to find out how many",
#         seed=42,
#         max_new_tokens=100,
#     )
#
#     assert (
#         response.generated_text
#         == " people died in the 1918 flu pandemic. Estimating the death toll of the 1918 flu pandemic is difficult because of incomplete records and because of the fact that many of the extra deaths were not attributed to the flu. Many experts believe that the 1918 flu pandemic killed between 50 and 100 million people. Iassistant\n\nThe 1918 flu pandemic, also known as the Spanish flu, is indeed one of the most devastating public health crises in human history. Estimating the exact"
#     )
#     assert response.details.generated_tokens == 100
#     assert response == response_snapshot
#
#
# async def test_flash_llama4_image_cow_dog(flash_llama4, response_snapshot):
#     image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
#     response = await flash_llama4.chat(
#         seed=42,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": image_url}},
#                     {
#                         "type": "text",
#                         "text": "What is the breed of the dog in the image?",
#                     },
#                 ],
#             },
#         ],
#         max_tokens=100,
#     )
#
#     assert (
#         response.choices[0].message.content
#         == "The image does not depict a dog; it shows a cow standing on a beach. Therefore, there is no breed of a dog to identify."
#     )
#     assert response.usage["completion_tokens"] == 30
#     assert response == response_snapshot
#
#
# async def test_flash_llama4_image_cow(flash_llama4, response_snapshot):
#     image_url = "https://huggingface.co/datasets/hf-internal-testing/fixtures-captioning/resolve/main/cow_beach_1.png"
#     response = await flash_llama4.chat(
#         seed=42,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": image_url}},
#                     {"type": "text", "text": "What is shown in this image?"},
#                 ],
#             },
#         ],
#         max_tokens=100,
#     )
#     assert (
#         response.choices[0].message.content
#         == "The image shows a brown cow standing on the beach with a white face and black and white marking on its ears. The cow has a white patch around its nose and mouth. The ocean and blue sky are in the background."
#     )
#     assert response.usage["completion_tokens"] == 46
#     assert response == response_snapshot
#
#
# # Helper function to convert a Pillow image to a base64 data URL
# def image_to_data_url(img: Image.Image, fmt: str) -> str:
#     buffer = BytesIO()
#     img.save(buffer, format=fmt)
#     img_data = buffer.getvalue()
#     b64_str = base64.b64encode(img_data).decode("utf-8")
#     mime_type = "image/png" if fmt.upper() == "PNG" else "image/jpeg"
#     return f"data:{mime_type};base64,{b64_str}"
#
#
# async def test_flash_llama4_image_base64_rgba(flash_llama4, response_snapshot):
#     # Create an empty 100x100 PNG image with alpha (transparent background)
#     img = Image.new("RGBA", (100, 100), (0, 0, 0, 0))
#     data_url = image_to_data_url(img, "PNG")
#     response = await flash_llama4.chat(
#         seed=42,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": data_url}},
#                     {
#                         "type": "text",
#                         "text": "What do you see in this transparent image?",
#                     },
#                 ],
#             },
#         ],
#         max_tokens=100,
#     )
#     assert response == response_snapshot
#
#
# async def test_flash_llama4_image_base64_rgb_png(flash_llama4, response_snapshot):
#     # Create an empty 100x100 PNG image without alpha (white background)
#     img = Image.new("RGB", (100, 100), (255, 255, 255))
#     data_url = image_to_data_url(img, "PNG")
#     response = await flash_llama4.chat(
#         seed=42,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": data_url}},
#                     {"type": "text", "text": "What do you see in this plain image?"},
#                 ],
#             },
#         ],
#         max_tokens=100,
#     )
#     assert response == response_snapshot
#
#
# async def test_flash_llama4_image_base64_rgb_jpg(flash_llama4, response_snapshot):
#     # Create an empty 100x100 JPEG image (white background)
#     img = Image.new("RGB", (100, 100), (255, 255, 255))
#     data_url = image_to_data_url(img, "JPEG")
#     response = await flash_llama4.chat(
#         seed=42,
#         messages=[
#             {
#                 "role": "user",
#                 "content": [
#                     {"type": "image_url", "image_url": {"url": data_url}},
#                     {"type": "text", "text": "What do you see in this JPEG image?"},
#                 ],
#             },
#         ],
#         max_tokens=100,
#     )
#     assert response == response_snapshot
