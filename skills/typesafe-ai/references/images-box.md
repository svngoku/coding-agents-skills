# Experimental Images and Box

Snapshot: 2026-10-25. The official pages label these features **"Dreamer-ready."**
They are not ordinary text examples or silently enabled defaults.

## Capability and permission gates

Before using vision:

1. Obtain authorization for the data and experimental feature.
2. Read current images, Box, model, and chosen SDK/API documentation.
3. Verify that the actual interface exposes assets and the necessary headers.
4. Validate and minimize the image locally before transmission.
5. Define absence handling and downstream authorization independently.

The example MCP judge inspected for this snapshot does not expose assets or Box.
Do not send unsupported fields to it, modify the MCP server without permission,
or infer SDK support from a broad tool schema.

Both image inputs and Box requests require the documented header
`TypeSafe-Danger-Zone-Images: true`. Images are documented for Jev 1.13/1.14;
Box requires 1.14. Recheck current support before use.

Sources: [images](https://docs.typesafe.ai/images.md),
[Box](https://docs.typesafe.ai/primitives/box.md),
[models](https://docs.typesafe.ai/models.md).

## Image constraints

The researched contract specifies:

| Constraint | Snapshot requirement |
| --- | --- |
| Formats | PNG/JPEG still images; no animated PNG |
| Encoded image size | At most 16 MiB per image |
| Width and height | Each 280–1568 pixels |
| Pixel count | 78,400–2,458,624 |
| Additional request limits | 32 MiB aggregate/expanded limits |

The API rejects invalid dimensions rather than automatically resizing them.
Apply EXIF orientation yourself before stripping metadata; stripping metadata
does not orient pixels. Evaluate the effects of resizing and cropping.

Do not copy the OpenAPI 16x16 example or Python assets 64x64 example as valid
current image fixtures: both conflict with the researched minimum dimensions.

## Assets and expansion

Use the current SDK's image helpers or documented HTTP asset format rather than
inventing a string marker. Exact construction belongs to the selected interface.

- Reference strings can occur in state, instructions, and criteria values,
  including inside larger strings, but not object keys.
- Each repeated reference expands again and consumes limits again.
- Unused assets fail validation.
- Multipart's first part is named `request`.
- Image parts need unique, nonempty names and each belongs to exactly one asset.
- Account for both compressed/decompressed request limits and expanded assets.

Source: [Python asset reference](https://docs.typesafe.ai/sdk/python/api/types/assets.md).

## Box is location, not existence

A Box response contains normalized `top`, `bottom`, `left`, and `right` edges.
They are fractions, not pixel coordinates. Convert against the image dimensions
and verify coordinate ordering/bounds before using them.

**A box can be returned when the target is absent.** Ask a separate presence Noul
and define what should happen when presence is uncertain or negative.
Do not draw, crop, click, or act merely because coordinates exist.

The response schema and SDK allow null confidence even where prose suggests a
number. Do not invent a universal Box confidence threshold. Evaluate localization,
presence detection, and any resulting action separately.

## Verification

- Check actual dimensions, orientation, format, size, and reference expansion.
- Include absent targets, multiple plausible targets, occlusion, and crop changes.
- Verify null-confidence handling and normalized-to-pixel conversion.
- Test the correct model/header/interface combination.
- Keep optical localization separate from permission to interact with a UI.

This skill ships no live image fixture or automatic upload. Its ordinary offline
checker intentionally covers only a conservative text-example profile.
