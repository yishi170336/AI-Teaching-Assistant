import { useMemo, useState } from 'react'
import { Button, Empty, Input, Modal, Popconfirm, Select, Spin, Tag, message } from 'antd'
import { BarChart3, CheckCircle2, ChevronRight, FilePlus2, Pencil, RefreshCw, Send, ShieldCheck, Trash2, UserRound, XCircle } from 'lucide-react'
import {
  createTeacherLearningReport,
  deleteTeacherLearningReport,
  fetchTeacherLearningReport,
  Homework,
  HomeworkLearningReport,
  HomeworkLearningReportSummary,
  runTeacherLearningReportAction,
  StudentProfile,
  updateTeacherStudent,
} from '../lib/api'
import HomeworkLearningReportDocument from './HomeworkLearningReportDocument'

const reportStatus: Record<HomeworkLearningReportSummary['status'], { label: string; color: string }> = {
  pending: { label: '等待生成', color: 'default' },
  generating: { label: '生成中', color: 'processing' },
  draft: { label: '待教师发布', color: 'blue' },
  published: { label: '已发布', color: 'success' },
  failed: { label: '生成失败', color: 'error' },
  blocked: { label: '质量门阻断', color: 'warning' },
}

export default function TeacherLearningAnalytics({
  students,
  reports,
  homeworks,
  loading,
  onRefresh,
}: {
  students: StudentProfile[]
  reports: HomeworkLearningReportSummary[]
  homeworks: Homework[]
  loading: boolean
  onRefresh: () => Promise<void>
}) {
  const [studentId, setStudentId] = useState(students[0]?.student_id || '')
  const [selectedReport, setSelectedReport] = useState<HomeworkLearningReport | null>(null)
  const [openingReportId, setOpeningReportId] = useState('')
  const [actionId, setActionId] = useState('')
  const [renameOpen, setRenameOpen] = useState(false)
  const [displayName, setDisplayName] = useState('')
  const [createOpen, setCreateOpen] = useState(false)
  const [selectedSubmissionIds, setSelectedSubmissionIds] = useState<string[]>([])
  const [reportTitle, setReportTitle] = useState('')

  const activeStudent = students.find((student) => student.student_id === studentId) || students[0]
  const activeStudentId = activeStudent?.student_id || ''
  const studentReports = reports.filter((report) => report.student_id === activeStudentId)
  const eligibleSubmissions = useMemo(() => homeworks.flatMap((homework) =>
    (homework.submissions || [])
      .filter((submission) => submission.student_id === activeStudentId && submission.status === 'graded' && submission.review?.passed)
      .map((submission) => ({
        value: submission.id,
        label: `${homework.title} · ${submission.grading?.total_score || 0}/${submission.grading?.max_score || 0} 分`,
      })),
  ), [activeStudentId, homeworks])

  const openReport = async (reportId: string) => {
    setOpeningReportId(reportId)
    try {
      setSelectedReport(await fetchTeacherLearningReport(reportId))
    } catch (error) {
      message.error(error instanceof Error ? error.message : '报告读取失败')
    } finally {
      setOpeningReportId('')
    }
  }

  const runAction = async (reportId: string, action: 'retry' | 'publish' | 'withdraw' | 'delete') => {
    setActionId(reportId)
    try {
      if (action === 'delete') await deleteTeacherLearningReport(reportId)
      else await runTeacherLearningReportAction(reportId, action)
      message.success(action === 'publish' ? '报告已发布给学生' : action === 'withdraw' ? '报告已撤回' : action === 'retry' ? '已重新开始生成' : '报告已删除')
      if (selectedReport?.id === reportId) setSelectedReport(null)
      await onRefresh()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '报告操作失败')
    } finally {
      setActionId('')
    }
  }

  const renameStudent = async () => {
    if (!activeStudentId || !displayName.trim()) return
    setActionId(activeStudentId)
    try {
      await updateTeacherStudent(activeStudentId, displayName)
      message.success('学生显示名已更新')
      setRenameOpen(false)
      await onRefresh()
    } catch (error) {
      message.error(error instanceof Error ? error.message : '学生姓名保存失败')
    } finally {
      setActionId('')
    }
  }

  const createStageReport = async () => {
    if (!activeStudentId || selectedSubmissionIds.length < 2) {
      message.warning('请至少选择两份已复核通过的作业')
      return
    }
    setActionId('create-report')
    try {
      const created = await createTeacherLearningReport({
        studentId: activeStudentId,
        submissionIds: selectedSubmissionIds,
        title: reportTitle,
      })
      message.success('累计报告已进入生成队列')
      setCreateOpen(false)
      setSelectedSubmissionIds([])
      setReportTitle('')
      await onRefresh()
      void openReport(created.id)
    } catch (error) {
      message.error(error instanceof Error ? error.message : '累计报告创建失败')
    } finally {
      setActionId('')
    }
  }

  if (loading) return <div className="teacher-loading"><Spin /><span>正在读取学生学情…</span></div>
  if (!students.length) return <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="学生提交作业后会自动建立学情档案" />

  return (
    <section className="teacher-learning-workspace">
      <aside className="teacher-student-roster">
        <header><div><span>STUDENTS</span><h2>学生档案</h2></div><Tag>{students.length} 人</Tag></header>
        <div className="teacher-student-list">
          {students.map((student) => (
            <button type="button" className={student.student_id === activeStudentId ? 'active' : ''} key={student.student_id} onClick={() => setStudentId(student.student_id)}>
              <span><UserRound size={17} /></span>
              <div><strong>{student.display_name}</strong><small>{student.graded_count} 份已批改 · 平均 {student.average_score_rate == null ? '—' : `${Math.round(student.average_score_rate * 100)}%`}</small></div>
              <ChevronRight size={15} />
            </button>
          ))}
        </div>
      </aside>

      <main className="teacher-learning-main">
        <header className="teacher-learning-student-header">
          <div><span><UserRound size={18} /></span><div><h2>{activeStudent.display_name}</h2><p>{activeStudent.submission_count} 份提交 · {activeStudent.pending_count} 份待处理</p></div></div>
          <div>
            <Button icon={<Pencil size={14} />} onClick={() => { setDisplayName(activeStudent.display_name); setRenameOpen(true) }}>修改显示名</Button>
            <Button type="primary" icon={<FilePlus2 size={14} />} disabled={eligibleSubmissions.length < 2} onClick={() => setCreateOpen(true)}>生成累计报告</Button>
          </div>
        </header>

        <div className="teacher-learning-summary-grid">
          <article><BarChart3 size={20} /><div><strong>{activeStudent.average_score_rate == null ? '—' : `${Math.round(activeStudent.average_score_rate * 100)}%`}</strong><span>平均得分率</span></div></article>
          <article><ShieldCheck size={20} /><div><strong>{studentReports.filter((report) => report.status === 'draft').length}</strong><span>待发布报告</span></div></article>
          <article><CheckCircle2 size={20} /><div><strong>{studentReports.filter((report) => report.status === 'published').length}</strong><span>已发布报告</span></div></article>
        </div>

        <section className="teacher-report-list-section">
          <header><div><span>LEARNING REPORTS</span><h3>单次与累计学情报告</h3></div><Button icon={<RefreshCw size={14} />} onClick={() => void onRefresh()}>刷新</Button></header>
          {studentReports.length ? <div className="teacher-report-list">{studentReports.map((report) => {
            const status = reportStatus[report.status]
            return (
              <article key={report.id}>
                <button type="button" onClick={() => void openReport(report.id)}>
                  <span className="teacher-report-icon"><BarChart3 size={18} /></span>
                  <div><strong>{report.title}</strong><small>{report.report_type === 'assignment' ? '单次作业' : `${report.metrics.assignment_count} 份作业累计`} · 得分率 {Math.round((report.metrics.score_rate || 0) * 100)}%</small></div>
                  {openingReportId === report.id ? <Spin size="small" /> : <Tag color={status.color}>{status.label}</Tag>}
                </button>
                <div className="teacher-report-card-actions">
                  {report.status === 'draft' && report.quality_status === 'passed' && <Button size="small" type="primary" icon={<Send size={13} />} loading={actionId === report.id} onClick={() => void runAction(report.id, 'publish')}>发布</Button>}
                  {report.status === 'published' && <Button size="small" icon={<XCircle size={13} />} loading={actionId === report.id} onClick={() => void runAction(report.id, 'withdraw')}>撤回</Button>}
                  {(report.status === 'failed' || report.status === 'blocked' || report.status === 'draft') && <Button size="small" icon={<RefreshCw size={13} />} loading={actionId === report.id} onClick={() => void runAction(report.id, 'retry')}>重新生成</Button>}
                  <Popconfirm title="删除这份学情报告？" okText="删除" cancelText="取消" okButtonProps={{ danger: true }} onConfirm={() => void runAction(report.id, 'delete')}><Button danger type="text" size="small" icon={<Trash2 size={13} />} /></Popconfirm>
                </div>
                {report.processing_error && <p>{report.processing_error}</p>}
              </article>
            )
          })}</div> : <Empty image={Empty.PRESENTED_IMAGE_SIMPLE} description="复核通过作业后会自动生成单次报告" />}
        </section>
      </main>

      <Modal open={renameOpen} title="修改学生显示名" okText="保存" cancelText="取消" confirmLoading={actionId === activeStudentId} onOk={() => void renameStudent()} onCancel={() => setRenameOpen(false)}>
        <Input value={displayName} maxLength={80} onChange={(event) => setDisplayName(event.target.value)} placeholder="学生显示名" />
      </Modal>
      <Modal open={createOpen} title="生成累计学情报告" okText="开始生成" cancelText="取消" confirmLoading={actionId === 'create-report'} onOk={() => void createStageReport()} onCancel={() => setCreateOpen(false)}>
        <div className="teacher-stage-report-form">
          <label><span>报告标题（可选）</span><Input value={reportTitle} maxLength={120} onChange={(event) => setReportTitle(event.target.value)} /></label>
          <label><span>选择已复核作业（至少两份）</span><Select mode="multiple" value={selectedSubmissionIds} options={eligibleSubmissions} onChange={setSelectedSubmissionIds} placeholder="选择作业" /></label>
        </div>
      </Modal>
      <Modal open={Boolean(selectedReport)} footer={null} width={1040} className="learning-report-modal" destroyOnHidden onCancel={() => setSelectedReport(null)}>
        {selectedReport && <HomeworkLearningReportDocument report={selectedReport} studentName={activeStudent.display_name} />}
      </Modal>
    </section>
  )
}
