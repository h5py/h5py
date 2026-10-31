Bug fixes
---------

* The standalone LZF filter example now closes and reopens the dataset before
  reading it, so its round-trip check exercises decompression instead of reading
  uncompressed data from the chunk cache.
