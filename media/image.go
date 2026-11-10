package media

import (
	"bytes"
	"fmt"
	"image"

	// the image formats DecodeImage reads; an application registers more
	// with image.RegisterFormat
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/tiff"
	_ "golang.org/x/image/webp"
)

// MaxImagePixels is the largest picture DecodeImage decodes: 256
// megapixels, a gigabyte of RGBA, which is beyond any camera and well short
// of what a crafted one-kilobyte PNG can claim to be.
const MaxImagePixels = 256 << 20

// DecodeImage decodes one picture in any registered format the way
// search.MediaIndex does, so that a picture hashed or embedded on its own
// (a phash.Hash next to an indexed vector, say) is the same picture the
// index saw. It refuses one that claims more than MaxImagePixels before any
// pixel is allocated, and turns a JPEG upright by its EXIF orientation,
// which image/jpeg ignores: a phone stores a portrait photo as landscape
// pixels plus that tag.
func DecodeImage(data []byte) (image.Image, error) {
	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if cfg.Width <= 0 || cfg.Height <= 0 || int64(cfg.Width)*int64(cfg.Height) > MaxImagePixels {
		return nil, fmt.Errorf("media: image is %dx%d pixels, more than %d", cfg.Width, cfg.Height, MaxImagePixels)
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	if format == "jpeg" {
		img = orient(img, jpegOrientation(data))
	}
	return img, nil
}
