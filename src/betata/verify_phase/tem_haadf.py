""" """

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

if __name__ == "__main__":
    """ """

    image_folder = Path(__file__).parents[3] / "data/verify_phase"
    image_name = "20250416_HAADF_1445_5.70_Mx_Wiener_Filtered.jpg"
    image_path = image_folder / image_name
    inset_name = "20250416_HAADF_1445_5.70_Mx_FFT.jpg"
    inset_path = image_folder / inset_name

    image = Image.open(image_path)
    # let both original and cropped images be squares
    image_dim = image.size[0]
    crop_dim = 1000
    image_crop_ratio = crop_dim / image_dim
    crop_x, crop_y = 720, 300
    crop_box = (crop_x, crop_y, crop_x + crop_dim, crop_y + crop_dim)
    cropped_image = image.crop(crop_box)

    image_with_scale_bar = ImageDraw.Draw(cropped_image)

    # 2 nm : 117 pixels in uncropped image
    scale_bar_pixel_width = 117 / image_crop_ratio
    scale_bar_physical_length = "2 nm"  # label
    scale_bar_color = (255, 255, 255)  # white
    scale_bar_thickness = 10
    scale_bar_x_pos = 50
    scale_bar_y_ofs = 35
    scale_bar_y_pos = cropped_image.height - scale_bar_y_ofs

    image_with_scale_bar.line(
        (
            scale_bar_x_pos,
            scale_bar_y_pos,
            scale_bar_x_pos + scale_bar_pixel_width,
            scale_bar_y_pos,
        ),
        fill=scale_bar_color,
        width=scale_bar_thickness,
    )

    # optional: add scale bar label
    #font_size = 75
    #try:
    #    font = ImageFont.truetype("/System/Library/Fonts/Avenir.ttc", size=font_size)
    #except IOError:
    #    font = ImageFont.load_default(size=font_size)

    #text_y_pos_offset = font_size * 1.2
    #text_y_pos = scale_bar_y_pos - text_y_pos_offset
    #image_with_scale_bar.text(
    #    (scale_bar_x_pos + font_size * 0.5, text_y_pos),
    #    scale_bar_physical_length,
    #    fill=scale_bar_color,
    #    font=font,
    #)

    image_save_path = Path(__file__).parents[3] / "out/verify_phase/TEM_HAADF.png"

    inset = Image.open(inset_path)
    inset_cx, inset_cy = inset.size[0] / 2, inset.size[1] / 2
    inset_dim = 200
    cropped_inset = inset.crop(
        (
            inset_cx - inset_dim / 2,
            inset_cy - inset_dim / 2,
            inset_cx + inset_dim / 2,
            inset_cy + inset_dim / 2,
        )
    )
    inset_scale_factor = 1.6
    scaled_inset = cropped_inset.resize(
        (
            int(cropped_inset.width * inset_scale_factor),
            int(cropped_inset.height * inset_scale_factor),
        )
    )
    inset_pos_x = cropped_image.width - scaled_inset.width
    inset_pos_y = cropped_image.height - scaled_inset.height
    inset_pos = (inset_pos_x - 25, inset_pos_y - scale_bar_y_ofs)

    cropped_image.paste(scaled_inset, inset_pos)
    cropped_image.save(image_save_path, dpi=(600, 600))

    cropped_image.show()
