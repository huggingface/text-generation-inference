module.exports = async ({
                            github,
                            context,
                            core
                        }) => {
    const owner = context.repo.owner;
    const repo = context.repo.repo;

    const workflows = await github.rest.actions.listRepoWorkflows({
        owner,
        repo
    })

    const workflow = workflows.data.workflows.find(w => w.path.includes(process.env.WORKFLOW_FILENAME));

    if (!workflow) {
        core.setFailed("No workflow found");
        return;
    }

    const runs = await github.rest.actions.listWorkflowRuns({
        owner,
        repo,
        workflow_id: workflow.id,
        status: "success",
        per_page: 1
    })

    if (runs.data.total_count === 0) {
        core.setFailed("No runs found");
        return;
    }

    const lastRelease = await github.rest.repos.getLatestRelease({
        owner,
        repo
    });

    const lastReleaseTag = lastRelease.data.tag_name;
    const tagRef = `tags/${lastReleaseTag}`;
    const lastReleaseCommit = await github.rest.git.getRef({
        owner,
        repo,
        ref: tagRef
    });
    const lastReleaseSha = lastReleaseCommit.data.object.sha;
    const lastReleaseRun = await github.rest.actions.listWorkflowRuns({
        owner,
        repo,
        workflow_id: workflow.id,
        head_sha: lastReleaseSha,
        status: "success",
        per_page: 1
    });
    let lastReleaseArtifacts = {data: {artifacts: []}};
    if (lastReleaseRun.data.total_count > 0) {
        lastReleaseArtifacts = await github.rest.actions.listWorkflowRunArtifacts({
            owner,
            repo,
            run_id: lastReleaseRun.data.workflow_runs[0].id
        });
    }

    const lastArtifacts = await github.rest.actions.listWorkflowRunArtifacts({
        owner,
        repo,
        run_id: runs.data.workflow_runs[0].id
    });

    const lastReleaseArtifact = lastReleaseArtifacts.data.artifacts.find(artifact => artifact.name === process.env.ARTIFACT_NAME);
    const lastArtifact = lastArtifacts.data.artifacts.find(artifact => artifact.name === process.env.ARTIFACT_NAME);

    if (lastReleaseArtifact) {
        await downloadArtifact(github, owner, repo, lastReleaseArtifact, lastReleaseTag);
    } else {
        console.log("No release artifact found")
    }
    if (lastArtifact) {
        await downloadArtifact(github, owner, repo, lastArtifact, lastArtifact.workflow_run.head_sha);
    } else {
        console.log("No last run artifact found")
    }
}

async function downloadArtifact(github, owner, repo, artifact, suffix) {
    const response = await github.rest.actions.downloadArtifact({
        owner,
        repo,
        artifact_id: artifact.id,
        archive_format: 'zip'
    });
    require('fs').writeFileSync(process.env.ARTIFACT_FILENAME, Buffer.from(response.data));
    // create directory to unzip
    require('fs').mkdirSync(`${process.env.UNZIP_DIR}/${artifact.workflow_run.head_sha}`, {recursive: true});
    require('child_process').execSync(`unzip -o ${process.env.ARTIFACT_FILENAME} -d ${process.env.UNZIP_DIR}/${suffix}`);

    console.log(`Artifact ${process.env.ARTIFACT_FILENAME} for ${suffix} downloaded successfully`);
}