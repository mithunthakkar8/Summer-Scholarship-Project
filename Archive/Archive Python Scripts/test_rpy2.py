# .venv\Scripts\activate

import rpy2.situation as S
print("Detected R version:", S.r_version_from_subprocess())

from rpy2.robjects import r
print(r('R.version.string'))

# print(r('.libPaths()'))

# # Add your personal R library path
# r('.libPaths("C:/Users/mithu/AppData/Local/R/win-library/4.5")')

# # Verify
# print(r('.libPaths()'))

r('library(lavaan)')
print(r('packageVersion("lavaan")'))

r('sessionInfo()')


