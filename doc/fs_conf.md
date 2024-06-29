# JSON Configuration Documentation

- **[Commons](#Commons)**
  - **[Region](#Region)**
  - **[Flip](#Flip)**
  - **[HSL](#HSL)**
  - **[BGR](#BGR)**
- **[GUI (GTK)](#GUI-GTK)**
  - **[GuiFrame](#GuiFrame)**
  - **[Gui](#Gui)**
- **[Miscellaneous](#Miscellaneous)**
  - **[Misc](#Misc)**
- **[Calibration](#Calibration)**
  - **[Pattern](#Pattern)**
  - **[Intrinsic](#intrinsic)**
  - **[Chain](#chain)**
  - **[Calibration](#calibration-1)**
  - **[ChainCalibration](#chaincalibration)**
  - **[Compose](#compose)**
- **[Filters](#Filters)**
  - **[Chroma](#chroma)**
  - **[Blur](#blur)**
  - **[Difference](#difference)**
    - **[BgKernelType](#bgkerneltype)**
- **[Pose](#Pose)**
  - **[PoseRoi](#poseroi)**
  - **[PoseThreshold](#posethresholds)**
  - **[PoseFilter](#posefilter)**
  - **[PoseModel](#posemodel)**
    - **[ModelBody](#modelbody)**
    - **[ModelDetector](#modeldetector)**
  - **[PoseUndistort](#poseundistort)**
  - **[PoseDevice](#posedevice)**
  - **[Pose](#pose-1)**
- **[Capture Device](#Device-Capture)**
  - **[Capture](#capture)**
- **[FULL EXAMPLE](#FULL-JSON-EXAMPLE)**
  - **[JsonConfig](#jsonconfig)**
    - **[ConfigType](#configtype)**

## Commons
### Region
- **Type:** Object

  | Property | Type      | Description  |
  |----------|-----------|--------------|
  | x        | `integer` | X-coordinate |
  | y        | `integer` | Y-coordinate |
  | w        | `integer` | Width        |
  | h        | `integer` | Height       |

- **Example:**
  ```json
  {
    "x": 10,
    "y": 20,
    "w": 200,
    "h": 100
  }
  ```

<br/>

### Flip
- **Type:** Object

  | Property | Type      | Description     |
  |----------|-----------|-----------------|
  | x        | `boolean` | Horizontal flip |
  | y        | `boolean` | Vertical flip   |

- **Example:**
  ```json
  {
    "x": true,
    "y": false
  }
  ```
  
<br/>

### HSL
- **Type:** Object

  | Property | Type    | Description               |
  |----------|---------|---------------------------|
  | h        | `float` | Hue. Range: [0..1]        |
  | s        | `float` | Saturation. Range: [0..1] |
  | l        | `float` | Lightness. Range: [0..1]  |

- **Example:**
  ```json
  {
    "h": 0.5,
    "s": 0.7,
    "l": 0.6
  }
  ```

<br/>


### BGR
- **Type:** Object

  | Property | Type    | Description          |
  |----------|---------|----------------------|
  | b        | `float` | Blue. Range: [0..1]  |
  | g        | `float` | Green. Range: [0..1] |
  | r        | `float` | Red. Range: [0..1]   |

- **Example:**
  ```json
  {
    "b": 0.2,
    "g": 0.4,
    "r": 0.6
  }
  ```
  
<br/>


## GUI (GTK)
### GuiFrame
- **Type:** Object

  | Property | Type      | Description       |
  |----------|-----------|-------------------|
  | w        | `integer` | Gui window width  |
  | h        | `integer` | Gui window height |

- **Example:**
  ```json
  {
    "w": 800,
    "h": 600
  }
  ```

<br/>

### Gui
- **Type:** Object

  | Property | Type                    | Description                          |
  |----------|-------------------------|--------------------------------------|
  | layout   | `integer[]`             | Layout of gui grid                   |
  | frame    | [`GuiFrame`](#guiframe) | Gui frame properties                 |
  | vertical | `boolean`               | Force vertical layout (optional)     |
  | scale    | `float`                 | Gui window scaling                   |
  | fps      | `integer`               | Gui render FPS limit                 |

- **Example:**
  ```json
  {
    "layout": [2, 3],
    "frame": {
      "w": 800,
      "h": 600
    },
    "vertical": true,
    "scale": 1.0,
    "fps": 30
  }
  ```

<br/>

## Miscellaneous
### Misc
- **Type:** Object

  | Property      | Type      | Description                           |
  |---------------|-----------|---------------------------------------|
  | capture_dummy | `boolean` | Use dummy source of frames            |
  | capture_fast  | `boolean` | Use faster method of frames retrieval |
  | debug         | `boolean` | Debug mode                            |
  | cpu           | `integer` | Default number of CPU cores available |

- **Example:**
  ```json
  {
    "capture_dummy": false,
    "capture_fast": false,
    "debug": false,
    "cpu": 8
  }
  ```

<br/>

## Calibration
### Pattern
- **Type:** Object

  | Property | Type                                      | Description                              |
  |----------|-------------------------------------------|------------------------------------------|
  | type     | `"radon"` \|  `"chessboard"` \| `"plain"` | Calibration pattern                      |
  | columns  | `integer`                                 | Number of columns in calibration pattern |
  | rows     | `integer`                                 | Number of rows in calibration pattern    |
  | size     | `float`                                   | Size of the calibration pattern square   |

- **Example:**
  ```json
  {
    "type": 1,
    "columns": 8,
    "rows": 6,
    "size": 2.5
  }
  ```
  
<br/>


### Intrinsic
- **Type:** Object

  | Property | Type      | Description                                                                |
  |----------|-----------|----------------------------------------------------------------------------|
  | x        | `float`   | X-coordinate                                                               |
  | y        | `float`   | Y-coordinate                                                               |
  | fix      | `boolean` | Fix camera device intrinsics (Don't try to optimise it during calibration) |

- **Example:**
  ```json
  {
    "x": 1.2,
    "y": 3.4,
    "fix": true
  }
  ```

<br/>


### Intrinsics
- **Type:** Object

  | Property | Type                      | Description                |
  |----------|---------------------------|----------------------------|
  | f        | [`Intrinsic`](#intrinsic) | Camera focal length values |
  | c        | [`Intrinsic`](#intrinsic) | Camera center position     |

- **Example:**
  ```json
  {
    "f": {
      "x": 1.2,
      "y": 3.4,
      "fix": true
    },
    "c": {
      "x": 5.6,
      "y": 7.8,
      "fix": false
    }
  }
  ```

<br/>


### Chain
- **Type:** Object

  | Property     | Type       | Description                                           |
  |--------------|------------|-------------------------------------------------------|
  | intrinsics   | `string[]` | List of file names with devices' intrinsic parameters |
  | closed       | `boolean`  | Is calibration chain closed                           |

- **Example:**
  ```json
  {
    "intrinsics": ["calib_1.json", "calib_2.json", "calib_3.json"],
    "closed": true
  }
  ```
  
<br/>


### Calibration
- **Type:** Object

  | Property   | Type                        | Description                                                          |
  |------------|-----------------------------|----------------------------------------------------------------------|
  | name       | `string`                    | Calibration session name                                             |
  | intrinsics | [`Intrinsics`](#intrinsics) | Calibration intrinsic properties                                     |
  | pattern    | [`Pattern`](#pattern)       | Calibration pattern properties                                       |
  | chain      | [`Chain`](#chain)           | Calibration chain properties                                         |
  | total      | `integer`                   | Total frames used in calibration process                             |
  | delay      | `integer`                   | Delay between consecutive frame shots in calibration process (in ms) |

- **Example:**
  ```json
  {
    "name": "camera_c1",
    "intrinsics": {
      "f": {
        "x": 1.2,
        "y": 3.4,
        "fix": true
      },
      "c": {
        "x": 5.6,
        "y": 7.8,
        "fix": false
      }
    },
    "pattern": {
      "type": 0,
      "columns": 9,
      "rows": 6,
      "size": 2.5
    },
    "chain": {
      "intrinsics": ["calib_1.json", "calib_2.json", "calib_3.json"],
      "closed": true
    },
    "total": 100,
    "delay": 5000
  }
  ```

<br/>


### ChainCalibration
- **Type:** Object

  | Property | Type       | Description                                           |
  |----------|------------|-------------------------------------------------------|
  | files    | `string[]` | List of file names with devices' intrinsic properties |
  | closed   | `boolean`  | Is calibration chain closed                           |

- **Example:**
  ```json
  {
    "files": ["calib_1.json", "calib_2.json", "calib_3.json"],
    "closed": true
  }
  ```
  
<br/>

### Compose
- **Type:** Object

  | Property | Type       | Description                     |
  |----------|------------|---------------------------------|
  | name     | `string`   | Calibration compose output name |
  | chain    | `string[]` | Chain calibration config        |

- **Example:**
  ```json
  {
    "name": "composed_calibration",
    "chain": ["calib_1.json", "calib_2.json", "calib_3.json"]
  }
  ```
  
<br/>

## Filters
### Chroma
- **Type:** Object

  | Property | Type          | Description                                                |
  |----------|---------------|------------------------------------------------------------|
  | key      | `string`      | Chromakey key color (hex)                                  |
  | replace  | `string`      | Chromakey replacement color (hex)                          |
  | range    | [`HSL`](#hsl) | HSL range (similarity threshold)                           |
  | blur     | `integer`     | Blur intensity `(CxC): C = (blur * 2) + 1`                 |
  | power    | `integer`     | Mask size (multiple of 256) `(TxT): T = (1 + power) * 256` |
  | fine     | `integer`     | Mask refinement kernel                                     |
  | refine   | `integer`     | Mask refinement iterations                                 |
  | linear   | `boolean`     | Use linear interpolation                                   |

- **Example:**
  ```json
  {
    "key": "#ffffff",
    "replace": "#000000",
    "range": {
      "h": 0.5,
      "s": 0.7,
      "l": 0.6
    },
    "blur": 5,
    "power": 256,
    "fine": 3,
    "refine": 2,
    "linear": true
  }
  ```
  
<br/>


### Blur
- **Type:** Object

  | Property | Type      | Description                                                      |
  |----------|-----------|------------------------------------------------------------------|
  | blur     | `integer` | Property for calculating kernel size `(CxC): C = (blur * 2) + 1` |

- **Example:**
  ```json
  {
    "blur": 5
  }
  ```
  
<br/>

### BgKernelType
- **Type:** Enum

  | Name       | Value |
  |------------|-------|
  | CROSS_4    | `1`   | 
  | SQUARE_8   | `2`   |
  | RUBY_12    | `3`   |
  | DIAMOND_16 | `4`   |

### Difference
- **Type:** Object

  | Property        | Type                            | Description                                      |
  |-----------------|---------------------------------|--------------------------------------------------|
  | BASE_RESOLUTION | `integer`                       | Segmentation mask base resolution (px)           |
  | color           | `string`                        | New background color (hex), ie: `"#ffffff"`      |
  | debug_on        | `boolean`                       | Enable debug functions                           |
  | adapt_on        | `boolean`                       | Enable updates of background model               |
  | ghost_on        | `boolean`                       | Enable "ghost" detection                         |
  | lbsp_on         | `boolean`                       | Use LBSP for spatial comparison                  |
  | norm_l2         | `boolean`                       | Use L2 distance for color comparison             |
  | linear          | `boolean`                       | Use linear interpolation for image downscaling   |
  | color_0         | `float`                         | Threshold for color comparison                   |
  | lbsp_0          | `float`                         | Threshold for LBSP comparison                    |
  | lbsp_d          | `float`                         | Threshold for LBSP calculation                   |
  | n_matches       | `integer`                       | Number of intersections for background detection |
  | t_upper         | `integer`                       | Maximal value of T(x)                            |
  | t_lower         | `integer`                       | Minimal value of T(x)                            |
  | model_size      | `integer`                       | Number of frames in background model             |
  | ghost_l         | `integer`                       | Temporary new T(x) value for "ghost" pixels      |
  | ghost_n         | `integer`                       | Number of frames for ghost classification        |
  | ghost_n_inc     | `integer`                       | Increment value for ghost_n                      |
  | ghost_n_dec     | `integer`                       | Decrement value for ghost_n                      |
  | alpha_d_min     | `float`                         | Constant learning rate for D_min(x)              |
  | alpha_norm      | `float`                         | Mixing alpha for dt(x) calculation               |
  | ghost_t         | `float`                         | Ghost threshold for local variations             |
  | r_scale         | `float`                         | Scale for R(x) feedback change                   |
  | r_cap           | `float`                         | Max value for R(x)                               |
  | t_scale_inc     | `float`                         | Scale for T(x) feedback increment                |
  | t_scale_dec     | `float`                         | Scale for T(x) feedback decrement                |
  | v_flicker_inc   | `float`                         | Increment v(x) value for flickering pixels       |
  | v_flicker_dec   | `float`                         | Decrement v(x) value for flickering pixels       |
  | v_flicker_cap   | `float`                         | Max value for v(x)                               |
  | refine_gate     | `integer`                       | Number of gate operations                        |
  | refine_erode    | `integer`                       | Number of erosion operations                     |
  | refine_dilate   | `integer`                       | Number of dilation operations                    |
  | gate_threshold  | `float`                         | Gate operation threshold                         |
  | kernel          | [`BgKernelType`](#bgkerneltype) | Background kernel type                           |
  | gate_kernel     | [`BgKernelType`](#bgkerneltype) | Gate kernel type                                 |
  | erode_kernel    | [`BgKernelType`](#bgkerneltype) | Erode kernel type                                |
  | dilate_kernel   | [`BgKernelType`](#bgkerneltype) | Dilate kernel type                               |

- **Example:**
  ```json
  {
    "BASE_RESOLUTION": 240,
    "color": "#ffffff",
    "debug_on": true,
    "adapt_on": true,
    "ghost_on": true,
    "lbsp_on": true,
    "norm_l2": true,
    "linear": false,
    "color_0": 0.032,
    "lbsp_0": 0.06,
    "lbsp_d": 0.025,
    "n_matches": 2,
    "t_upper": 256,
    "t_lower": 2,
    "model_size": 50,
    "ghost_l": 2,
    "ghost_n": 300,
    "ghost_n_inc": 1,
    "ghost_n_dec": 15,
    "alpha_d_min": 0.75,
    "alpha_norm": 0.75,
    "ghost_t": 0.25,
    "r_scale": 0.1,
    "r_cap": 255,
    "t_scale_inc": 0.5,
    "t_scale_dec": 0.25,
    "v_flicker_inc": 1.0,
    "v_flicker_dec": 0.1,
    "v_flicker_cap": 255,
    "refine_gate": 0,
    "refine_erode": 0,
    "refine_dilate": 0,
    "gate_threshold": 0.85,
    "kernel": 4,
    "gate_kernel": 4,
    "erode_kernel": 3,
    "dilate_kernel": 3
  }
  ```

<br/>


### Filter
- **Type:** Object

  | Property                                | Type                                                                  | Description |
  |-----------------------------------------|-----------------------------------------------------------------------|-------------|
  | type                                    | `"blur"` \| `"chromakey"` \| `"difference"`                           | Filter type |
  | `{ ...blur, ...chroma, ...difference }` | [`Blur`](#blur) \| [`Chroma`](#chroma) \| [`Difference`](#difference) | Properties  |

- **Example:**
  ```json
  {
    "type": "chromakey",
    "key": "#ffffff",
    "replace": "#000000",
    "range": {
    "h": 0.5,
    "s": 0.7,
    "l": 0.6
    },
    "blur": 5,
    "power": 256
  }
  ```

  ```json
  {
    "type": "blur",
    "blur": 3
  }
  ```

  ```json
  {
    "type": "difference",
    "BASE_RESOLUTION": 240,
    "color": "#ffffff"
  }
  ```

<br/>

## Pose
### PoseRoi
- **Type:** Object

  | Property        | Type    | Description                                                                  |
  |-----------------|---------|------------------------------------------------------------------------------|
  | rollback_window | `float` | Distance between detectors and actual ROI middle point. Range: [0.0 ... 1.0] |
  | center_window   | `float` | Distance between actual and predicted ROI middle point. Range: [0.0 ... 1.0] |
  | clamp_window    | `float` | Acceptable ratio of clamped to original ROI size. Range: [0.0 ... 1.0]       |
  | scale           | `float` | Scaling factor for ROI                                                       |
  | margin          | `float` | Margins added to ROI                                                         |
  | padding_x       | `float` | Horizontal paddings added to ROI                                             |
  | padding_y       | `float` | Vertical paddings added to ROI                                               |

- **Example:**
  ```json
  {
    "rollback_window": 0.2,
    "center_window": 0.1,
    "clamp_window": 0.5,
    "scale": 1.5,
    "margin": 0.1,
    "padding_x": 0.05,
    "padding_y": 0.05
  }
  ```

<br/>


### PoseThresholds
- **Type:** Object

  | Property | Type    | Description                                                                   |
  |----------|---------|-------------------------------------------------------------------------------|
  | detector | `float` | Threshold score for detector ROI presence. Range: [0.0 ... 1.0]               |
  | marks    | `float` | Threshold score for landmarks presence. Range: [0.0 ... 1.0]                  |
  | pose     | `float` | Threshold score for pose presence. Range: [0.0 ... 1.0]                       |
  | roi      | `float` | Threshold score for detector ROI distance to body marks. Range: [0.0 ... 1.0] |

- **Example:**
  ```json
  {
    "detector": 0.8,
    "marks": 0.7,
    "pose": 0.9,
    "roi": 0.6
  }
  ```

<br/>

### PoseFilter
- **Type:** Object

  | Property | Type      | Description                    |
  |----------|-----------|--------------------------------|
  | velocity | `float`   | Low-pass filter velocity scale |
  | window   | `integer` | Low-pass filter window size    |
  | fps      | `integer` | Low-pass filter target FPS     |

- **Example:**
  ```json
  {
    "velocity": 0.5,
    "window": 10,
    "fps": 30
  }
  ```

<br/>

### ModelBody
- **Type:** Enum

  | Name         | Value         | Description               |
  |--------------|---------------|---------------------------|
  | HEAVY_ORIGIN | `"heavy"`     | Original heavy model      |
  | FULL_ORIGIN  | `"full"`      | original full model       |
  | LITE_ORIGIN  | `"lite"`      | original lite model       |
  | HEAVY_F32    | `"heavy_f32"` | F32 quantized heavy model |
  | FULL_F32     | `"full_f32"`  | F32 quantized full model  |
  | LITE_F32     | `"lite_f32"`  | F32 quantized lite model  |
  | HEAVY_F16    | `"heavy_f16"` | F16 quantized heavy model |
  | FULL_F16     | `"full_f16"`  | F16 quantized full model  |
  | LITE_F16     | `"lite_f16"`  | F16 quantized lite model  |

<br/>

### ModelDetector
- **Type:** Enum

  | Name   | Value      | Description                  |
  |--------|------------|------------------------------|
  | ORIGIN | `"origin"` | Original detector model      |
  | F_32   | `"f_32"`   | F32 quantized detector model |
  | F_16   | `"f_16"`   | F16 quantized detector model |

<br/>

### PoseModel
- **Type:** Object

  | Property | Type                              | Description              |
  |----------|-----------------------------------|--------------------------|
  | detector | [`ModelDetector`](#modeldetector) | BlazePose detector model |
  | body     | [`ModelBody`](#modelbody)         | BlazePose body model     |

- **Example:**
  ```json
  {
    "detector": 1,
    "body": 3
  }
  ```

<br/>


### PoseUndistort
- **Type:** Object

  | Property | Type      | Description                                  |
  |----------|-----------|----------------------------------------------|
  | source   | `boolean` | Undistort input image                        |
  | points   | `boolean` | Undistort position of localized points       |
  | alpha    | `float`   | Free scaling parameter. Range: [0.0 ... 1.0] |

- **Example:**
  ```json
  {
    "source": true,
    "points": false,
    "alpha": 0.5
  }
  ```
  
<br/>


### PoseDevice
- **Type:** Object

  | Property   | Type                                | Description                                       |
  |------------|-------------------------------------|---------------------------------------------------|
  | intrinsics | `string`                            | Name of the file with device intrinsic parameters |
  | threshold  | [`PoseThresholds`](#posethresholds) | Pose thresholds                                   |
  | undistort  | [`PoseUndistort`](#poseundistort)   | Pose undistort properties                         |
  | filter     | [`PoseFilter`](#posefilter)         | Pose filter properties                            |
  | model      | [`PoseModel`](#posemodel)           | Pose model properties                             |
  | roi        | [`PoseRoi`](#poseroi)               | Pose ROI properties                               |

- **Example:**
  ```json
  {
    "intrinsics": "camera_1_calib.json",
    "threshold": {
      "detector": 0.8,
      "marks": 0.7,
      "pose": 0.9,
      "roi": 0.6
    },
    "undistort": {
      "source": true,
      "points": false,
      "alpha": 0.5
    },
    "filter": {
      "velocity": 0.5,
      "window": 10,
      "fps": 30
    },
    "model": {
      "detector": "f32",
      "body": "full_f32"
    },
    "roi": {
      "rollback_window": 0.2,
      "center_window": 0.1,
      "clamp_window": 0.5,
      "scale": 1.5,
      "margin": 0.1,
      "padding_x": 0.05,
      "padding_y": 0.05
    }
  }
  ```

<br/>


### Pose
- **Type:** Object

  | Property      | Type                                    | Description                                |
  |---------------|-----------------------------------------|--------------------------------------------|
  | devices       | [`PoseDevice[]`](#posedevice)           | Array of capture devices                   |
  | chain         | [`ChainCalibration`](#chaincalibration) | Chain calibration config                   |
  | show_epilines | `boolean`                               | Show epipolar lines (debug)                |
  | segmentation  | `boolean`                               | Perform segmentation                       |
  | threads       | `integer`                               | Number of dedicated CPU threads (optional) |

- **Example:**
  ```json
  {
    "devices": [
      {
        "intrinsics": "camera1_calib.json",
        "threshold": {
          "detector": 0.8,
          "marks": 0.7,
          "pose": 0.9,
          "roi": 0.6
        },
        "undistort": {
          "source": true,
          "points": false,
          "alpha": 0.5
        },
        "filter": {
          "velocity": 0.5,
          "window": 10,
          "fps": 30
        },
        "model": {
          "detector": "f32",
          "body": "full_f32"
        },
        "roi": {
          "rollback_window": 0.2,
          "center_window": 0.1,
          "clamp_window": 0.5,
          "scale": 1.5,
          "margin": 0.1,
          "padding_x": 0.05,
          "padding_y": 0.05
        }
      }
    ],
    "chain": {
      "files": [
        "calib_1.json", 
        "calib_2.json", 
        "calib_3.json"
      ],
      "closed": false
    },
    "show_epilines": false,
    "segmentation": false,
    "threads": 8
  }
  ```
  
<br/>

## Device Capture
### Capture
- **Type:** Object

  | Property | Type                  | Description                                                            |
  |----------|-----------------------|------------------------------------------------------------------------|
  | id       | `string`              | Capture device id. i.e:  `"/dev/video1"` or `"usb-0000:02:00.0-2"` etc |
  | name     | `string`              | Capture device name (arbitrary)                                        |
  | codec    | `string`              | Capture codec. i.e: `"MJPG"` \| `"YUYV"` \| `"H264"` \| `"BGR3"`       |
  | width    | `integer`             | Captured frame width                                                   |
  | height   | `integer`             | Captured frame height                                                  |
  | buffer   | `integer`             | Capture buffer size. May introduce lag, smoothes fps.                  |
  | fps      | `integer`             | Capture FPS                                                            |
  | rotate   | `boolean`             | Frame rotation (optional). Applied after `flip`                        |
  | flip     | [`Flip`](#flip)       | Frame flip (optional). Applied after `region`                          |
  | region   | [`Region`](#region)   | Frame subregion (optional)                                             |
  | filters  | [`Filter[]`](#filter) | Array of filters (optional)                                            |

- **Example:**
  ```json
  {
    "id": "/dev/video1",
    "name": "Camera 1",
    "codec": "MJPG",
    "width": 1920,
    "height": 1080,
    "buffer": 3,
    "fps": 30,
    "region": {
      "x": 10,
      "y": 20,
      "w": 800,
      "h": 600
    },
    "flip": {
      "x": false,
      "y": true
    },
    "rotate": true,
    "filters": [
      {
        "type": "blur",
        "blur": 5
      },
      {
        "type": "chromakey",
        "key": "#ffffff",
        "replace": "#000000",
        "range": {
          "h": 0.5,
          "s": 0.7,
          "l": 0.6
        },
        "blur": 5,
        "power": 256,
        "fine": 3,
        "refine": 2,
        "linear": true
      },
      {
        "type": "difference",
        "color": "#ffffff"
      }
    ]
  }
  ```

<br/>

## FULL JSON EXAMPLE

### ConfigType
- **Type:** Enum

  | Name              | Value                 | Description                                |
  |-------------------|-----------------------|--------------------------------------------|
  | CALIBRATION       | `"calibration"`       | Single camera calibration                  |
  | CHAIN_CALIBRATION | `"chain_calibration"` | Chain calibration                          |
  | CROSS_CALIBRATION | `"cross_calibration"` | Cross calibration                          |
  | COMPOSE           | `"compose"`           | Compose calibration chain into single pair |
  | POSE              | `"pose"`              | Motion capture pose estimation             |

### JsonConfig
- **Type:** Object

  | Property    | Type                            | Description                       |
  |-------------|---------------------------------|-----------------------------------|
  | type        | [`ConfigType`](#configtype)     | Configuration type                |
  | misc        | [`Misc`](#misc)                 | Miscellaneous configuration       |
  | gui         | [`Gui`](#gui)                   | Gui configuration                 |
  | pose        | [`Pose`](#pose)                 | Motion capture configuration      |
  | captures    | [`Capture[]`](#capture)         | Array of capturing devices        |
  | calibration | [`Calibration`](#calibration-1) | Calibration configuration         |
  | compose     | [`Compose`](#compose)           | Calibration compose configuration |

- **Example:**
  ```json
  {
    "type": 4,
    "misc": {
      "capture_dummy": false,
      "capture_fast": false,
      "debug": true,
      "cpu": 8
    },
    "gui": {
      "layout": [2, 3],
      "frame": {
        "w": 800,
        "h": 600
      },
      "vertical": true,
      "scale": 1.0,
      "fps": 300
    },
    "pose": {
      "devices": [
        {
          "intrinsics": "camera_1_calib.json",
          "threshold": {
            "detector": 0.8,
            "marks": 0.7,
            "pose": 0.9,
            "roi": 0.6
          },
          "undistort": {
            "source": true,
            "points": false,
            "alpha": 0.5
          },
          "filter": {
            "velocity": 0.5,
            "window": 10,
            "fps": 30
          },
          "model": {
            "detector": "f32",
            "body": "full_f32"
          },
          "roi": {
            "rollback_window": 0.2,
            "center_window": 0.1,
            "clamp_window": 0.5,
            "scale": 1.5,
            "margin": 0.1,
            "padding_x": 0.05,
            "padding_y": 0.05
          }
        }
      ],
      "chain": {
        "files": ["calib_1.json", "calib_2.json", "calib_3.json"],
        "closed": false
      },
      "show_epilines": true,
      "segmentation": true,
      "threads": 8
    },
    "captures": [
      {
        "id": "/dev/video1",
        "name": "Camera 1",
        "codec": "MJPG",
        "width": 1920,
        "height": 1080,
        "buffer": 3,
        "fps": 30,
        "flip": {
          "x": false,
          "y": true
        },
        "region": {
          "x": 10,
          "y": 20,
          "w": 800,
          "h": 600
        },
        "rotate": true,
        "filters": [
          {
            "type": "blur",
            "blur": 5
          },
          {
            "type": "chromakey",
            "key": "#ffffff",
            "replace": "#000000",
            "range": {
              "h": 0.5,
              "s": 0.7,
              "l": 0.6
            },
            "blur": 5,
            "power": 256,
            "fine": 3,
            "refine": 2,
            "linear": true
          },
          {
            "type": "difference",
            "BASE_RESOLUTION": 240,
            "color": "#ffffff",
            "debug_on": true,
            "adapt_on": true,
            "ghost_on": true,
            "lbsp_on": true,
            "norm_l2": true,
            "linear": false,
            "color_0": 0.032,
            "lbsp_0": 0.06,
            "lbsp_d": 0.025,
            "n_matches": 2,
            "t_upper": 256,
            "t_lower": 2,
            "model_size": 50,
            "ghost_l": 2,
            "ghost_n": 300,
            "ghost_n_inc": 1,
            "ghost_n_dec": 15,
            "alpha_d_min": 0.75,
            "alpha_norm": 0.75,
            "ghost_t": 0.25,
            "r_scale": 0.1,
            "r_cap": 255,
            "t_scale_inc": 0.5,
            "t_scale_dec": 0.25,
            "v_flicker_inc": 1.0,
            "v_flicker_dec": 0.1,
            "v_flicker_cap": 255,
            "refine_gate": 0,
            "refine_erode": 0,
            "refine_dilate": 0,
            "gate_threshold": 0.85,
            "kernel": 4,
            "gate_kernel": 4,
            "erode_kernel": 3,
            "dilate_kernel": 3
          }
        ]
      }
    ],
    "calibration": {
      "name": "calibration1",
      "intrinsics": {
        "f": {
          "x": 1.2,
          "y": 3.4,
          "fix": true
        },
        "c": {
          "x": 5.6,
          "y": 7.8,
          "fix": false
        }
      },
      "pattern": {
        "type": 0,
        "columns": 9,
        "rows": 6,
        "size": 2.5
      },
      "chain": {
        "intrinsics": ["calib_1.json", "calib_2.json", "calib_3.json"],
        "closed": false
      },
      "total": 100,
      "delay": 500
    },
    "compose": {
      "name": "calib_1_3.json",
      "chain": ["calib_1.json", "calib_2.json", "calib_3.json"]
    }
  }
  ```

