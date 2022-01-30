# Change log
## 1.6.1 - 2022-01-29
### Added
- Runs with Python V3
- Support for output to textfile 

## 1.3.0 - 2016-12-12
### Added
- Drawing of up to 6 waveforms from multichannel audio files


## 1.2.0 - 2016-12-11
### Fixed
- Sanitize output filename


## 1.1.1 - 2016-09-09
### Fixed
- Make output object from spotidump similar to that from ffmpeg


## 1.1.0 - 2016-04-15
### Added
- Option to enable overview generation
- Option to generate overview per directory or for all analysed files


## 1.0.1 - 2016-04-13
### Changed
- Use socket.io instead of long poll
- More styling and error handling in web app


## 1.0.0 - 2016-04-12
### Added
- Web app / web GUI


## 0.12.0 - 2016-04-02
### Added
- Draw all channels in all graphs except for main waveform display


## 0.11.1 - 2016-04-02
### Changed
- Disable some debug print statements


## 0.11.0 - 2016-04-02
### Added
- Option to render overview image
- Support for separate output directory
- Display total track RMS and Crest
### Changed
- LFE channel color changed to yellow from black


## 0.10.4 - 2016-03-23
### Fixed
- Ignore files without audio tracks


## 0.10.3 - 2016-03-22
### Changed
- Use argparse instead of optparse
- Use proper logging facility for debugging messages
### Fixed
- Make all ffmpeg tag names lowercase


## 0.10.2 - 2016-03-13
### Changed
- Track metadata bps value as int instead of string
### Fixed
- Ignore some divide by zero errors


## 0.10.1 - 2016-03-12
### Changed
- Reshape integer array directly from buffer


## 0.10.0 - 2016-03-12
### Added
- Manual
- Support for 24 bit and multi channel files
### Changed
- Pipe output from ffmpeg to memory buffer instead of temp file


## 0.9.7 - 2016-01-11
### Added
- Support for mono tracks
- Display Peak Loudness Range (PLR)
### Changed
- Layout


## 0.9.6 - 2016-01-11
### Added
- Calculate True Peak of track and per channel
- Display DR value


## 0.9.5 - 2016-01-08
### Added
- Calculate Dynamic Range (DR) value
### Changed
- Tweak short term loudness graph


## 0.9.3 - 2016-01-07
### Added
- Support for EBU R 128 loudness calculations


## 0.9.2 - 2015-10-17
### Fixed
- Use floats when calculating downsampled data Fs


## 0.9.1 - 2015-10-17
### Added
- Support for analysing multiple files


## 0.9.0 - 2015-10-15
### Changed
- Make SpotiDump use new pyspotify lib
- Use track name as filename when analysing Spotify tracks
### Fixed
- Limit length of displayed album name in header


## 0.8.0 - 2015-10-14
### Added
- Metadata shown in header
### Changed
- Make graph layout more closely resemble MasVis
- Reduce output image bit depth to decrease file size
### Fixed
- Prevent tick overlap of axis end labels
### Removed
- GUI


## 0.7.0 - 2015-10-13
### Changed
- Only import Spotify helper when needed
- Draw histogram data on top of helper lines


## 0.6.0 - 2013-03-21
### Added
- Support for Spotify


## 0.5.0 - 2013-02-11
### Fixed
- Support non ASCII filenames in GUI
- Strip metadata from output when using ffmpeg


## 0.4.0 - 2013-02-09
### Added
- Include ffmpeg in app bundle
- App icon
- GUI progress bar
### Changed
- Make loudest span position more visible
- Change output filename suffix
### Fixed
- Display range of histogram


## 0.3.2 - 2013-02-08
### Fixed
- Filetype check


## 0.3.1 - 2013-02-08
### Fixed
- Calculate histogram using integer sample data


## 0.3.0 - 2013-02-08
### Added
- GUI
### Changed
- Graph axis styles
### Removed
- AudioLab dependency


## 0.2.0 - 2012-09-02
### Added
- Support for mp3 etc using ffmpeg
### Changed
- Fix data downsampling calculations


## 0.1.0 - 2012-09-01
### Changed
- Downsample data before plotting
- Use MasVis plot styles


## 0.0.3 - 2012-09-01
### Changed
- Calculate normalized average spectrum using LTAS


## 0.0.2 - 2012-08-31
### Changed
- Calculate real length of downsampled data when plotting


## 0.0.1 - 2012-08-31
### Added
- Initial release
- License
- Saving of plot to image file