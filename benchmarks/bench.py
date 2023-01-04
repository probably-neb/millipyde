import logging
logging.basicConfig(level = logging.INFO)

from benchers import millipyde_bench, scikit_image_bench
millipyde_bench.main()
scikit_image_bench.main()
