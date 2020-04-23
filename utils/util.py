import os

from PIL import Image, ImageDraw

"""
def draw_box(boxes, c='black'):
    # minX, minY, maxX, maxY
    for _, (minX, minY, maxX, maxY) in enumerate(boxes):
        plt.hlines(minY, minX, maxX, colors=c, lw=2)
        plt.hlines(maxY, minX, maxX, colors=c, lw=2)
        plt.vlines(minX, minY, maxY, colors=c, lw=2)
        plt.vlines(maxX, minY, maxY, colors=c, lw=2)
"""

def draw_box(image, boxes, result_dir, file_name):
    
    rect_image = ImageDraw.Draw(image)
    for box in boxes:
        rect_image.rectangle([(box[0], box[1]), (box[2], box[3])], fill=None, outline='red', width=2)
    
    image.save(os.path.join(result_dir, str(file_name) + '.png'))
def tensor_to_image(img_tensors):
    tensor = (img_tensors.clone()+1)*0.5 * 255
    tensor = tensor.cpu().clamp(0, 255)
    
    array = tensor.numpy().astype('uint8')
    if array.shape[0] == 1:
        array = array.squeeze(0)
    elif array.shape[0] == 3:
        array = array.swapaxes(0, 1).swapaxes(1, 2)
    print('arrayy', array.shape)
    return Image.fromarray(array)

def make_directory(*dirs):
    for directory in dirs:
        if not os.path.exists(directory):
            os.makedirs(directory)
            
            
    
