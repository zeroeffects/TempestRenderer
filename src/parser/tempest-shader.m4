# C M4 Macros for Bison.

# Copyright (C) 2002, 2004, 2005, 2006, 2007, 2008, 2009, 2010 Free
# Software Foundation, Inc.

# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

m4_include(b4_pkgdatadir/[c++.m4])

# b4_symbol_actions(FILENAME, LINENO,
#                   SYMBOL-TAG, SYMBOL-NUM,
#                   SYMBOL-ACTION, SYMBOL-TYPENAME)
# -------------------------------------------------
m4_define([b4_symbol_actions],
[m4_pushdef([b4_dollar_dollar],
   [m4_ifval([$6], [NodeT<$6>(std::move(*yyvaluep))], [std::move(*yyvaluep)])])dnl
m4_pushdef([b4_at_dollar], [(*yylocationp)])dnl
      case $4: /* $3 */
b4_syncline([$2], [$1])
	$5;
b4_syncline([@oline@], [@ofile@])
	break;
m4_popdef([b4_at_dollar])dnl
m4_popdef([b4_dollar_dollar])dnl
])

# b4_lhs_value([TYPE])
# --------------------
# Expansion of $<TYPE>$.
m4_define([b4_lhs_value],
[yyval])
# [m4_ifval([$1], [NodeBase<$1>(std::move(yyval))], [std::move(yyval)])])


# b4_rhs_value(RULE-LENGTH, NUM, [TYPE])
# --------------------------------------
# Expansion of $<TYPE>NUM, where the current rule has RULE-LENGTH
# symbols on RHS.
m4_define([b4_rhs_value],
[(m4_ifval([$3], [NodeT<$3>(std::move(yysemantic_stack_@{($1) - ($2)@}))], [std::move(yysemantic_stack_@{($1) - ($2)@})]))])
