/////////////////////
// Custom Components
///////////////////// 

#let frame(content) = [
  #set align(left)
  #set par(first-line-indent: 0em)
  #block(
    width: 100%,
    inset: 5pt,
    stroke: 0.5pt + black,    
  )[#content]
]