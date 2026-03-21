from PIL import Image, ImageDraw

map_folder = "/sim_ws/src/f1tenth_gym_ros/maps/"
input_file = "hallway.pgm"
output_file = "hallway_obstacle.pgm"
img = Image.open(map_folder + input_file).convert("L")  # force grayscale
draw = ImageDraw.Draw(img)

draw.rectangle([170, 360, 200, 380], fill=0)

draw.rectangle([230, 330, 250, 360], fill=0)

draw.rectangle([290, 350, 310, 360], fill=0)

img.save(map_folder + output_file)