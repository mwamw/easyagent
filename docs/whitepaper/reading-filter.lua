-- Short ASCII destinations avoid oversized PDF names for long CJK headings.
-- Pandoc builds the table of contents after applying the filter.
local heading_number = 0

function Header(header)
  heading_number = heading_number + 1
  header.identifier = string.format("section-%02d", heading_number)
  return header
end
