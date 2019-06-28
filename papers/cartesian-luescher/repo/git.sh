#!/usr/bin/env bash

OLD=$1
NEW=$2

THIS=$(realpath $0)
GIT=$(dirname ${THIS})/..

if [ -z "$OLD" ]; then
	OLD=HEAD
fi

if [ -z "$NEW" ]; then
	NEW="--"
fi

# Double hyphens is a ligature in TeX:
if [ "--" == "${NEW}" ]; then
	NEWPRINT="-{}-"
else
	NEWPRINT="${NEW}"
fi

# Handle commit IDs that have ~ in them, like HEAD~3
OLDPRINT=${OLD/\~/\{\\textasciitilde\}}
NEWPRINT=${NEWPRINT/\~/\{\\textasciitilde\}}

pushd ${GIT} 2>/dev/null 1>/dev/null

result="`git diff --name-only ${OLD} ${NEW} 2>/dev/null | wc -l | tr -d [:blank:]`"' files in '"\texttt{${NEWPRINT}}"' different from commit '"\texttt{${OLDPRINT}}"' from '"`git show -s --format=%ad --date=iso ${OLD}`"

# Correct "1 files" pluralization madness.
if [[ "${result:0:2}" == "1 " ]]; then
    result=${result/1 files/1 file}
fi

# Turn red if there are dirty files.
if [[ ! "${result:0:1}" == "0" ]]; then
    result="{\color{Red}${result}}"
else
    result="{\color{Green}${result}}"
fi

result="\newcommand{\repositoryInformationSetup}{
    \usepackage[dvipsnames]{xcolor}
    \usepackage[ angle=90, color=black, opacity=1, scale=2, ]{background} 
    \SetBgPosition{current page.west} 
    \SetBgVshift{-4.5mm} 
    \backgroundsetup{contents={${result}}}
}"

popd 2>/dev/null 1>/dev/null

echo "${result}"